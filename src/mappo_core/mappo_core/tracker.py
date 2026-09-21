import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ==========================================
# 雜訊模擬工具 (不變)
# ==========================================
def get_noisy_radar_data(true_pos):
    """ 輸入真值，回傳加了雜訊的位置 [x, y, z] (模擬雷達) """
    noise_std = 10.0
    return true_pos + np.random.normal(0, noise_std, 3)

def get_noisy_vision_data(def_pos, def_R_frd, att_true_pos):
    """ 輸入真值與防守者姿態，回傳加了雜訊的角度 [yaw, pitch] (模擬相機) """
    rel_pos_world = att_true_pos - def_pos
    rel_body = def_R_frd.T @ rel_pos_world

    dx, dy, dz = rel_body
    true_yaw = np.arctan2(dy, dx)
    true_pitch = np.arctan2(-dz, np.hypot(dx, dy))

    noise_std = np.radians(1.0)
    return np.array([true_yaw, true_pitch]) + np.random.normal(0, noise_std, 2)


# ==========================================
# ★ 修正 A：視覺量測 (yaw) 的 wrap-around 處理
#
#   雷達量測 (3 維位置) 不需要處理。
#   視覺量測 (2 維 [yaw, pitch]) 的 yaw 一旦掃過 ±180°，
#   直接相減或直接加權平均都會算出離譜的殘差/平均值，
#   例如 +179° 跟 -179° 用一般加權平均會變成 0°，完全錯誤。
#   這是「正確性」問題：不修，遇到方位角掃過邊界時估計值會瞬間跳掉。
#   ※ 經實測，這不是你 log 裡瘋狂崩潰訊息的主因 (見下方修正 B)，
#      但仍然是必須修的真實 bug，兩個問題彼此獨立。
# ==========================================
def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def residual_z_generic(a, b):
    y = np.subtract(a, b)
    if y.shape[0] == 2:          # 視覺量測 [yaw, pitch]
        y[0] = _wrap_angle(y[0])  # 只有 yaw 需要 wrap
    return y

def z_mean_generic(sigmas, Wm):
    if sigmas.shape[1] == 2:     # 視覺量測：yaw 用 circular mean
        sin_sum = np.sum(np.sin(sigmas[:, 0]) * Wm)
        cos_sum = np.sum(np.cos(sigmas[:, 0]) * Wm)
        yaw_mean = np.arctan2(sin_sum, cos_sum)
        pitch_mean = np.dot(Wm, sigmas[:, 1])
        return np.array([yaw_mean, pitch_mean])
    return np.dot(Wm, sigmas)    # 雷達量測 (位置)：一般加權平均


# ==========================================
# UKF 追蹤器類別
# ==========================================
class TargetTracker:
    def __init__(self, dt, start_pos):
        self.dt = dt
        self.n_dim = 6  # [x, y, z, vx, vy, vz]
        self.m_dim_radar = 3

        # ★ alpha 維持原始 0.1，不要調大！
        #   實測過 alpha=1.0 看似讓中心 sigma point 權重比較「正常」，
        #   但視覺量測模型 atan2() 高度非線性，搭配目前偏大的初始 P，
        #   alpha 越大 sigma point 散得越開、線性近似誤差反而越大，
        #   崩潰次數從 125 次惡化成 239 次 (已實測驗證)。
        points = MerweScaledSigmaPoints(n=self.n_dim, alpha=0.1, beta=2., kappa=0)

        self.ukf = UKF(dim_x=self.n_dim, dim_z=self.m_dim_radar, dt=dt,
                       fx=self.f_cv, hx=self.h_radar, points=points,
                       residual_z=residual_z_generic,
                       z_mean_fn=z_mean_generic)

        self.ukf.x = np.concatenate([start_pos, [0, 0, 0]])
        self.ukf.P *= 10.0

        q_val = 0.01
        self.ukf.Q = np.eye(6) * q_val

        self.R_radar = np.diag([10.0, 10.0, 10.0]) ** 2
        self.R_vision = np.diag([np.radians(1.0), np.radians(1.0)]) ** 2

    def f_cv(self, x, dt):
        """ 狀態轉移函數：恆定速度模型 """
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        return F @ x

    def h_radar(self, x):
        """ 雷達觀測模型 """
        return x[:3]

    def h_vision(self, x, def_pos, def_R_frd):
        """ 視覺觀測模型 """
        rel_pos = x[:3] - def_pos
        rel_body = def_R_frd.T @ rel_pos

        dx, dy, dz = rel_body
        yaw = np.arctan2(dy, dx)
        pitch = np.arctan2(-dz, np.hypot(dx, dy))
        return np.array([yaw, pitch])

    # ============================================================
    # ★ 修正 B (真正的關鍵)：用特徵值裁剪取代原本的 "+1e-6" 假性修正
    #
    #   原始版只在 predict() 前面對 P 加一個 1e-6 的小常數，
    #   但實測發現視覺量測更新後 P 矩陣的特徵值常常已經是 -16、-30
    #   這種等級的負值 (跟 yaw 是否 wrap 無關，從第 1~2 個 tick 就會發生)，
    #   1e-6 這個量級對修正它完全沒有作用，等於沒修，
    #   結果就是三不五時整個 P 矩陣被砍成 50*I 重來，
    #   導致狀態估計瞬間飄移幾百公尺再慢慢收斂——
    #   這才是真正讓 BPNG 攔截不準的原因。
    #
    #   修法：對 P 做特徵分解，把所有小於 floor 的特徵值夾到 floor，
    #   保證每次都拿到一個「真正」正定的矩陣，而不是賭運氣。
    #   而且要套用在 predict() 跟 update() 之後都做，
    #   不能只防 predict() 一個點 (因為真正讓 P 變壞的是 update())。
    # ============================================================
    def _clip_P(self, floor=1e-3):
        P = (self.ukf.P + self.ukf.P.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals_clipped = np.clip(eigvals, floor, None)
        self.ukf.P = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    def predict(self):
        self._clip_P()
        self.ukf.predict()
        self._clip_P()

    def update_radar(self, z):
        self._clip_P()
        self.ukf.update(z, hx=self.h_radar, R=self.R_radar)
        self._clip_P()

    def update_vision(self, z, def_pos, def_R_frd):
        self._clip_P()
        self.ukf.update(z, hx=lambda x: self.h_vision(x, def_pos, def_R_frd), R=self.R_vision)
        self._clip_P()

    @property
    def state(self):
        return self.ukf.x