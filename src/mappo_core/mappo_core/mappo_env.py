"""
mappo_env.py (Updated with Attacker Maneuvering)
===========================================================
MATLAB `mappo_env_step_reset_v2_static.m` 的 Python (NumPy) 1:1 移植版。

★ 新增：入侵者機動參數 (att_serpentine / att_evade / att_random_turn)
  - att_serp_period = 8.0 秒（蛇行周期，改大 = 更慢）
  - att_turn_period = 7.0 秒（轉向間隔，改大 = 更慢）
  - 其他參數見下方

設計目標：數值上盡可能與 MATLAB 完全一致，方便驗證 Python 訓練是否能
複現 MATLAB 結果。所有特徵順序、正規化係數 (除以 200)、t_hit 算法、
候選排序、reward 公式都嚴格照原檔。

對應關係：
  defaultConfig()              -> Config
  initEpisode()                -> Env.reset()
  stepEpisode()                -> Env.step()
  buildObsForAgent()           -> Env._build_obs_for_agent()
  buildCandidatesAllAgents()   -> Env._build_candidates()
  jammerStepFixed()            -> Env._jammer_step_fixed()
  幾何工具 (lineRectIntersection 等) -> 模組層級函式
===========================================================
"""

import numpy as np


# =====================================================================
# 參數設定 (對應 defaultConfig)
# =====================================================================
class Config:
    def __init__(self):
        self.cx, self.cy = 100.0, 80.0
        self.W, self.H = 70.0, 40.0
        self.dt = 0.05
        self.actionHoldSec = 1.0
        self.att_speed_range = np.array([1.5, 3.0])
        self.sensorRange = 70.0

        # 入侵者生成距離：加在 half_diag 之上的額外距離範圍 [min, max]
        # 對應原本 sample_attackers 寫死的 +40 / +160
        self.att_dist_range = np.array([40.0, 160.0])

        # PN
        self.PN_N = 4
        self.PN_overlapEps = 0.2
        self.PN_patrolReturn = True

        # jammer
        self.jammer_pos = np.array([self.cx, self.cy])
        self.jammer_range = 100.0
        self.jammer_cooldown = 1.0
        self.jammer_nextAvailableTime0 = 1.0
        self.jammer_p_succ_const = 0.1

        self.maxEpisodeSec = 200.0
        self.NattMin = 8
        self.NattMax = 12

        self.K_max = 5
        self.NO_OP = self.K_max  # 最後一個動作索引 = no-op
        self.Ndef = 4

        self.def_speed_min = 3.0
        self.def_speed_max = 5.0
        self.def_speed_default = 4.0

        # 防禦者動力學限制:對齊 PX4 預設值 (typhoon_h480 airframe 未覆寫這兩個參數)
        self.def_max_accel = 5.0                # m/s^2, PX4 MPC_ACC_HOR_MAX 預設
        self.pn_psi_dot_max = np.radians(45.0)  # rad/s, PX4 MPC_YAWRAUTO_MAX 預設 (45 deg/s)

        # reward
        self.r_step_time_pen = -0.02
        self.r_kill_reward = 100.0
        self.r_duplicate_penalty = -200.0
        self.r_invalid_action_pen = -10.0
        self.r_jam_reward = 20.0
        self.r_penetration_pen = -200.0
        self.r_success_weight = 200.0
        self.r_energy_weight = -0.006
        self.r_static_clear_bonus = 100.0
        
        # ===== 勢能塑形 (突防威脅) =====
        self.r_threat_shaping = True          # 啟用威脅塑形
        self.r_threat_weight = 0.5            # 塑形強度，需調整(0.3~0.8)
        self.threat_dist_scale = 50.0         # 距離正規化尺度(m)
        
        # ===== 閒置懲罰 =====
        self.r_idle_pen = -50.0                # 有活目標卻閒置的懲罰

        # ===== 入侵者機動參數 =====
        # 主開關：是否啟用機動
        self.att_maneuver = False
        
        # 隨機轉向 (Random Turn)
        self.att_random_turn = False
        self.att_turn_period = 5.0         # 轉向時間間隔 (秒)，改大 = 更慢
        self.att_turn_max_angle = 15.0     # 最大轉向角度 (度)
        
        # 逃避 (Evade)
        self.att_evade = False
        self.att_evade_radius = 25.0       # 逃避半徑 (m)
        self.att_evade_gain = 1.5          # 逃避增益


# =====================================================================
# 幾何工具 (對應 MATLAB 同名 helper)
# =====================================================================
def make_rect(cfg):
    rect = np.array([
        [cfg.cx - cfg.W / 2, cfg.cy - cfg.H / 2],
        [cfg.cx + cfg.W / 2, cfg.cy - cfg.H / 2],
        [cfg.cx + cfg.W / 2, cfg.cy + cfg.H / 2],
        [cfg.cx - cfg.W / 2, cfg.cy + cfg.H / 2],
    ])
    edge_len = np.linalg.norm(np.diff(np.vstack([rect, rect[0]]), axis=0), axis=1)
    perim = edge_len.sum()
    return rect, edge_len, perim


def line_intersect(p0, v, a, w):
    M = np.column_stack([v, -w])
    if abs(np.linalg.det(M)) < 1e-6:
        return np.inf, np.inf
    sol = np.linalg.solve(M, (a - p0))
    return sol[0], sol[1]


def line_rect_intersection(p0, v, rect):
    t_list, P_list, edges, u_list = [], [], [], []
    for k in range(4):
        A = rect[k]
        B = rect[(k + 1) % 4]
        t, u = line_intersect(p0, v, A, B - A)
        if np.isfinite(t) and t >= 0 and 0 <= u <= 1:
            t_list.append(t)
            P_list.append(p0 + t * v)
            edges.append(k)
            u_list.append(u)
    if not t_list:
        return np.inf, p0.copy(), 0, 0.0
    idx = int(np.argmin(t_list))
    return t_list[idx], P_list[idx], edges[idx], u_list[idx]


def position_on_perimeter(rect, edge_len, s):
    s = np.mod(s, edge_len.sum())
    for k in range(4):
        if s <= edge_len[k]:
            A = rect[k]
            B = rect[(k + 1) % 4]
            return A + (B - A) * (s / edge_len[k])
        s -= edge_len[k]
    return rect[0].copy()


def point_inside_rect(p, rect):
    xmin, xmax = rect[:, 0].min(), rect[:, 0].max()
    ymin, ymax = rect[:, 1].min(), rect[:, 1].max()
    return (xmin <= p[0] <= xmax) and (ymin <= p[1] <= ymax)


def _orient(a, b, c):
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def _on_seg(a, b, c):
    return (min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12 and
            min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12)


def segment_intersect(p1, p2, q1, q2):
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    if o1 == 0 and _on_seg(p1, q1, p2):
        return True
    if o2 == 0 and _on_seg(p1, q2, p2):
        return True
    if o3 == 0 and _on_seg(q1, p1, q2):
        return True
    if o4 == 0 and _on_seg(q1, p2, q2):
        return True
    return (o1 != o2) and (o3 != o4)


def segment_cross_rect_boundary(p0, p1, rect):
    for k in range(4):
        A = rect[k]
        B = rect[(k + 1) % 4]
        if segment_intersect(p0, p1, A, B):
            return True
    return False


def is_penetrated_now(p_prev, p_curr, rect):
    return segment_cross_rect_boundary(p_prev, p_curr, rect) or point_inside_rect(p_curr, rect)



# ===== 工具函數:威脅勢能 =====
def dist_point_to_rect_boundary(p, rect):
    """
    計算點 p 到矩形邊界的最短距離 (正數表示在矩形外)
    rect 格式: [[x_min, y_min], [x_max, y_max], ...]
    """
    x_min, y_min = rect[0]
    x_max, y_max = rect[1]
    x, y = p[0], p[1]
    
    # 計算 x、y 各軸到邊界的距離
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    
    # 歐式距離
    return np.sqrt(dx**2 + dy**2)

def sample_attackers(cfg, N, rng, stratified=False):
    """stratified=True 時將 360 度切成 N 個扇區各取一個角度，避免小 N 隨機聚在同一側。"""
    half_diag = 0.5 * np.sqrt(cfg.W ** 2 + cfg.H ** 2)
    r_min = half_diag + cfg.att_dist_range[0]
    r_max = half_diag + cfg.att_dist_range[1]
    att_pos = np.zeros((N, 2))
    sector = 2 * np.pi / N
    for i in range(N):
        r = r_min + (r_max - r_min) * rng.random()
        th = sector * i + sector * rng.random() if stratified else 2 * np.pi * rng.random()
        att_pos[i] = np.array([cfg.cx, cfg.cy]) + r * np.array([np.cos(th), np.sin(th)])
    return att_pos


def point_to_segment_dist(P, A, B):
    AB = B - A
    AP = P - A
    den = np.dot(AB, AB)
    if den < 1e-12:
        return np.linalg.norm(P - A)
    t = np.clip(np.dot(AP, AB) / den, 0.0, 1.0)
    proj = A + t * AB
    return np.linalg.norm(P - proj)


def wrap_to_pi(ang):
    return np.mod(ang + np.pi, 2 * np.pi) - np.pi


def pn_step_2d(pM, psi, psi_init, pT, vT, VM, Npn, dt, psi_dot_max):
    """比例導引一步。回傳 (新位置, 新psi, psi_init)。"""
    r = pT - pM
    R2 = r[0] ** 2 + r[1] ** 2
    if R2 < 1e-10:
        return pM, psi, psi_init
    if not psi_init:
        psi = np.arctan2(r[1], r[0])
        psi_init = True
    vM = VM * np.array([np.cos(psi), np.sin(psi)])
    vRel = vT - vM
    lambda_dot = (r[0] * vRel[1] - r[1] * vRel[0]) / R2
    psi_dot = np.clip(Npn * lambda_dot, -psi_dot_max, psi_dot_max)
    psi = wrap_to_pi(psi + psi_dot * dt)
    vM = VM * np.array([np.cos(psi), np.sin(psi)])
    pM = pM + vM * dt
    return pM, psi, psi_init


# =====================================================================
# 入侵者 / 防禦者 狀態容器
# =====================================================================
class Attacker:
    __slots__ = ['pos', 'dir', 'speed', 'state',
                 'maneuver_state_turn', 'maneuver_state_evade']

    def __init__(self):
        self.pos = np.zeros(2)
        self.dir = np.array([1.0, 0.0])
        self.speed = 0.0
        self.state = 'UNKNOWN'   # UNKNOWN / IMMUNE / JAMMED
        # 機動狀態追蹤
        self.maneuver_state_turn = {'t_next': 0.0}     # 下次轉向時間
        self.maneuver_state_evade = {'t_last': 0.0, 'target_dir': None}


class Defender:
    __slots__ = ['s', 'pos', 'sensor', 'mode', 'targetID', 'psi', 'psi_init', 'speed']

    def __init__(self):
        self.s = 0.0
        self.pos = np.zeros(2)
        self.sensor = 0.0
        self.mode = 'PATROL'      # PATROL / PN_CHASE
        self.targetID = -1        # 1-based 攻擊者 ID，-1 表示無
        self.psi = 0.0
        self.psi_init = False
        self.speed = 0.0


# =====================================================================
# 環境主體
# =====================================================================
class MappoEnv:
    """
    對應 MATLAB 環境。注意攻擊者 ID 在內部沿用 MATLAB 的 1-based 慣例
    (targetID, candidate 內容都是 1-based)，只有在「動作索引 -> 候選位置」
    這種陣列存取時才用 0-based，與原檔行為一致。
    """

    def __init__(self, cfg=None, seed=None):
        self.cfg = cfg if cfg is not None else Config()
        self.rng = np.random.default_rng(seed)
        # 由訓練端設定 (對應 global current_speed_range / current_static_mode)
        self.speed_range = None
        self.static_mode = False
        self.dist_range = None      # 入侵者生成距離範圍；None 或 [0,0] = 用環境預設
        self.natt_range = None      # ★ 入侵者數量範圍；None = 用 Config.NattMin/NattMax
        self.stratified_spawn = False  # ★ 入侵者生成角度是否分扇區以避免聚集 (預設關閉，維持原本訓練分佈)
        self.state = None

    # ----- 由訓練端呼叫，等同 MATLAB 在 reset 前設 global -----
    def configure(self, speed_range=None, static_mode=None, dist_range=None,
                  natt_range=None, attacker=None, stratified_spawn=None):
        if speed_range is not None:
            self.speed_range = np.asarray(speed_range, dtype=float)
        if static_mode is not None:
            self.static_mode = bool(static_mode)
        if dist_range is not None:
            dr = np.asarray(dist_range, dtype=float)
            # [0,0] 代表「此階段不覆寫距離，用環境預設」
            if np.allclose(dr, 0.0):
                self.dist_range = None
            else:
                self.dist_range = dr
        # ★ 新增：入侵者數量範圍 (課程訓練用)。
        if natt_range is not None:
            nr = np.asarray(natt_range, dtype=int)
            self.natt_range = (int(nr[0]), int(nr[1]))
        else:
            self.natt_range = None
        
        # ★ 新增：入侵者機動策略設定 (attacker 字典)
        if attacker is not None:
            for key, val in attacker.items():
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, val)

        if stratified_spawn is not None:
            self.stratified_spawn = bool(stratified_spawn)

    # =================================================================
    # reset  (對應 initEpisode + buildObs)
    # =================================================================
    def reset(self):
        cfg = self.cfg
        if self.speed_range is not None:
            cfg.att_speed_range = self.speed_range
        if self.dist_range is not None:
            cfg.att_dist_range = self.dist_range
        # ★ 套用課程訓練指定的 Natt 範圍 (不指定則用 Config 預設 [8,12])
        natt_min, natt_max = cfg.NattMin, cfg.NattMax
        if self.natt_range is not None:
            natt_min, natt_max = self.natt_range

        st = {}
        rect, edge_len, perim = make_rect(cfg)
        st['rect'] = rect
        st['edgeLen'] = edge_len
        st['perim'] = perim
        st['t'] = 0.0
        st['static_mode'] = self.static_mode

        Natt = int(self.rng.integers(natt_min, natt_max + 1))  # inclusive
        st['Natt'] = Natt
        att_pos = sample_attackers(cfg, Natt, self.rng, stratified=self.stratified_spawn)
        st['aliveAtt'] = np.ones(Natt, dtype=bool)

        st['att'] = [Attacker() for _ in range(Natt)]
        st['P_int'] = np.full((Natt, 2), np.nan)
        st['t_hit'] = np.full(Natt, np.nan)
        st['s_hit'] = np.full(Natt, np.nan)

        center = np.array([cfg.cx, cfg.cy])
        for i in range(Natt):
            st['att'][i].pos = att_pos[i].copy()
            dir_vec = center - att_pos[i]
            dir_vec = dir_vec / max(1e-9, np.linalg.norm(dir_vec))
            st['att'][i].dir = dir_vec

            if self.static_mode:
                st['att'][i].speed = 0.0
            else:
                st['att'][i].speed = (cfg.att_speed_range[0] +
                                      self.rng.random() *
                                      (cfg.att_speed_range[1] - cfg.att_speed_range[0]))
            st['att'][i].state = 'UNKNOWN'

            dist_hit, Pint, eIdx, u_i = line_rect_intersection(att_pos[i], dir_vec, rect)
            st['P_int'][i] = Pint
            # t_hit：靜止時用距離當代理時間，否則距離/速度
            if st['att'][i].speed < 1e-6:
                st['t_hit'][i] = dist_hit
            else:
                st['t_hit'][i] = dist_hit / st['att'][i].speed
            st['s_hit'][i] = edge_len[:eIdx].sum() + edge_len[eIdx] * u_i

        # 防禦者初始化
        st['Ndef'] = cfg.Ndef
        s0 = [40,
              cfg.W + cfg.H / 2,
              cfg.W + cfg.H + cfg.W / 2,
              cfg.W + cfg.H + cfg.W + cfg.H / 2]
        st['defs'] = [Defender() for _ in range(cfg.Ndef)]
        for d in range(cfg.Ndef):
            st['defs'][d].s = s0[d]
            st['defs'][d].pos = position_on_perimeter(rect, edge_len, s0[d])
            st['defs'][d].sensor = cfg.sensorRange
            st['defs'][d].mode = 'PATROL'
            st['defs'][d].targetID = -1
            st['defs'][d].psi = 0.0
            st['defs'][d].psi_init = False
            st['defs'][d].speed = cfg.def_speed_default

        st['aware'] = np.zeros((cfg.Ndef, Natt), dtype=bool)
        st['jammer_target'] = -1
        st['jammer_nextAvailableTime'] = cfg.jammer_nextAvailableTime0

        st['lastStepKills'] = np.zeros(cfg.Ndef)
        st['lastStepJams'] = 0
        st['lastStepPen'] = 0
        st['totalKills'] = 0
        st['totalJams'] = 0
        st['penetrated'] = 0
        st['prev_threat'] = 0.0           # ★ 初始化前一步威脅勢能

        self.state = st
        st['lastCandidates'] = self._build_candidates()
        obs_list = self._build_obs_all()
        return obs_list

    # =================================================================
    # 入侵者機動步進（在 step 中調用）
    # =================================================================
    def _step_attacker_heading(self, i):
        """更新入侵者 i 的方向，應用機動策略。"""
        cfg = self.cfg
        st = self.state
        att = st['att'][i]
        center = np.array([cfg.cx, cfg.cy])
        
        # 基礎目標方向：指向中心
        target_dir = center - att.pos
        target_dir_norm = np.linalg.norm(target_dir)
        if target_dir_norm > 1e-6:
            target_dir = target_dir / target_dir_norm
        
        # 不適用機動時直接用目標方向
        if not cfg.att_maneuver:
            att.dir = target_dir
            return
        
        # 隨機轉向
        if cfg.att_random_turn:
            if st['t'] >= att.maneuver_state_turn['t_next']:
                angle_change = (self.rng.random() - 0.5) * 2 * np.radians(cfg.att_turn_max_angle)
                cos_a, sin_a = np.cos(angle_change), np.sin(angle_change)
                target_dir = np.array([
                    cos_a * target_dir[0] - sin_a * target_dir[1],
                    sin_a * target_dir[0] + cos_a * target_dir[1]
                ])
                att.maneuver_state_turn['t_next'] = st['t'] + cfg.att_turn_period
        
        # 逃避
        if cfg.att_evade:
            closest_def = None
            min_dist = np.inf
            for d in range(cfg.Ndef):
                dist = np.linalg.norm(st['defs'][d].pos - att.pos)
                if dist < cfg.att_evade_radius and dist < min_dist:
                    min_dist = dist
                    closest_def = d
            
            if closest_def is not None:
                def_pos = st['defs'][closest_def].pos
                evade_dir = att.pos - def_pos
                evade_norm = np.linalg.norm(evade_dir)
                if evade_norm > 1e-6:
                    evade_dir = evade_dir / evade_norm
                    target_dir = target_dir + cfg.att_evade_gain * evade_dir
                    target_dir_norm = np.linalg.norm(target_dir)
                    if target_dir_norm > 1e-6:
                        target_dir = target_dir / target_dir_norm
        
        att.dir = target_dir

    # =================================================================
    # step  (對應 stepEpisode + buildObs)
    # =================================================================
    def step(self, target_actions, velocity_cmds):
        cfg = self.cfg
        st = self.state
        Ndef = cfg.Ndef

        st['lastStepKills'] = np.zeros(Ndef)
        st['lastStepJams'] = 0
        st['lastStepPen'] = 0
        r_indiv = np.zeros(Ndef)

        # 本次 actionHoldSec 期間要追的目標速度 (受 cfg.def_max_accel 限制,
        # 每個 dt 子迴圈才真正逼近一步,見下方內迴圈)
        v_cmd = np.clip(velocity_cmds, cfg.def_speed_min, cfg.def_speed_max)

        # ----- Step 0: 解析 target action -----
        candidates = st['lastCandidates']
        chosenIDs = -np.ones(Ndef, dtype=int)

        for d in range(Ndef):
            a = int(target_actions[d])

            if a == cfg.NO_OP:
                if st['defs'][d].targetID > 0:
                    tid = st['defs'][d].targetID
                    # tid 為 1-based -> 陣列以 tid-1 存取
                    if (tid < 1 or tid > st['Natt'] or
                            not st['aliveAtt'][tid - 1] or
                            st['att'][tid - 1].state != 'IMMUNE'):
                        st['defs'][d].mode = 'PATROL'
                        st['defs'][d].targetID = -1
                        st['defs'][d].psi_init = False
                chosenIDs[d] = st['defs'][d].targetID
                continue

            idx = a  # 0-based 候選索引 (MATLAB a+1 對應這裡 a)
            cand = candidates[d]
            if cand is None or idx >= len(cand) or cand[idx] <= 0:
                r_indiv[d] += cfg.r_invalid_action_pen
                chosenIDs[d] = st['defs'][d].targetID
                continue

            new_target = cand[idx]  # 1-based
            if (new_target < 1 or new_target > st['Natt'] or
                    not st['aliveAtt'][new_target - 1] or
                    st['att'][new_target - 1].state != 'IMMUNE'):
                r_indiv[d] += cfg.r_invalid_action_pen
                chosenIDs[d] = st['defs'][d].targetID
                continue

            if st['defs'][d].targetID != new_target:
                st['defs'][d].targetID = new_target
                st['defs'][d].mode = 'PN_CHASE'
                st['defs'][d].psi_init = False
            chosenIDs[d] = new_target

        # Duplicate penalty
        for d in range(Ndef):
            if chosenIDs[d] <= 0:
                continue
            others = np.concatenate([chosenIDs[:d], chosenIDs[d + 1:]])
            if np.any(others == chosenIDs[d]):
                r_indiv[d] += cfg.r_duplicate_penalty

        # ----- Step 1: 內部 dt 模擬迴圈 -----
        nInner = round(cfg.actionHoldSec / cfg.dt)
        isDone = False

        for _ in range(nInner):
            if st['t'] >= cfg.maxEpisodeSec:
                isDone = True
                break

            # 防禦者速度逼近 v_cmd (加速度受 cfg.def_max_accel 限制,每個 dt 更新一次)
            for d in range(Ndef):
                dv = np.clip(v_cmd[d] - st['defs'][d].speed,
                             -cfg.def_max_accel * cfg.dt, cfg.def_max_accel * cfg.dt)
                st['defs'][d].speed += dv

            # ★ 入侵者機動更新
            for i in range(st['Natt']):
                if st['aliveAtt'][i]:
                    self._step_attacker_heading(i)

            # 入侵者前進
            for i in range(st['Natt']):
                if st['aliveAtt'][i] and st['att'][i].speed > 1e-6:
                    p_prev = st['att'][i].pos.copy()
                    p_curr = p_prev + st['att'][i].dir * st['att'][i].speed * cfg.dt
                    st['att'][i].pos = p_curr
                    if is_penetrated_now(p_prev, p_curr, st['rect']):
                        st['penetrated'] += 1
                        st['lastStepPen'] += 1
                        st['aliveAtt'][i] = False

            # jammer
            self._jammer_step_fixed()

            # 感測 awareness
            for i in range(st['Natt']):
                if st['aliveAtt'][i] and st['att'][i].state == 'IMMUNE':
                    for d in range(Ndef):
                        if np.linalg.norm(st['defs'][d].pos - st['att'][i].pos) <= st['defs'][d].sensor:
                            st['aware'][d, i] = True

            # PN 追擊
            for d in range(Ndef):
                if st['defs'][d].mode == 'PN_CHASE' and st['defs'][d].targetID > 0:
                    i1 = st['defs'][d].targetID    # 1-based
                    i = i1 - 1
                    if (i1 < 1 or i1 > st['Natt'] or not st['aliveAtt'][i] or
                            st['att'][i].state != 'IMMUNE'):
                        st['defs'][d].mode = 'PATROL'
                        st['defs'][d].targetID = -1
                        st['defs'][d].psi_init = False
                    else:
                        p_prev = st['defs'][d].pos.copy()
                        vT = st['att'][i].dir * st['att'][i].speed  # 靜止時為 0
                        new_pos, new_psi, new_init = pn_step_2d(
                            st['defs'][d].pos, st['defs'][d].psi, st['defs'][d].psi_init,
                            st['att'][i].pos, vT, st['defs'][d].speed, cfg.PN_N, cfg.dt,
                            cfg.pn_psi_dot_max)
                        st['defs'][d].pos = new_pos
                        st['defs'][d].psi = new_psi
                        st['defs'][d].psi_init = new_init

                        p_curr = st['defs'][d].pos
                        dp = np.linalg.norm(p_curr - st['att'][i].pos)
                        dseg = point_to_segment_dist(st['att'][i].pos, p_prev, p_curr)
                        if dp <= cfg.PN_overlapEps or dseg <= cfg.PN_overlapEps:
                            st['aliveAtt'][i] = False
                            st['lastStepKills'][d] += 1
                            st['totalKills'] += 1
                            if cfg.PN_patrolReturn:
                                st['defs'][d].mode = 'PATROL'
                                st['defs'][d].targetID = -1
                                st['defs'][d].psi_init = False

            st['t'] += cfg.dt
            if not np.any(st['aliveAtt']):
                isDone = True
                break

        # ----- Step 2: Reward -----
        # 個體獎勵:擊殺 + 能量消耗
        for d in range(Ndef):
            r_indiv[d] += cfg.r_kill_reward * st['lastStepKills'][d]

        energy_terms = np.zeros(Ndef)
        for d in range(Ndef):
            energy_terms[d] = cfg.r_energy_weight * st['defs'][d].speed ** 2
            r_indiv[d] += energy_terms[d]

        # ===== 閒置懲罰 =====
        # 檢查:是否有沒被任何防守者指派的存活入侵者
        assigned = set(st['defs'][d].targetID for d in range(Ndef)
                       if st['defs'][d].targetID >= 0)
        unassigned_alive = [i for i in range(st['Natt'])
                            if st['aliveAtt'][i] and i not in assigned]
        
        if unassigned_alive:  # 還有沒人管的活目標
            for d in range(Ndef):
                is_idle = (st['defs'][d].targetID < 0)
                if is_idle:
                    r_indiv[d] += cfg.r_idle_pen  # 懲罰閒置

        # ===== 勢能塑形 (威脅衰減獎勵) =====
        threat_now = self._total_threat(st, cfg)
        if cfg.r_threat_shaping:
            threat_prev = st.get('prev_threat', threat_now)
            delta_threat = threat_prev - threat_now  # 威脅下降為正
            r_team_threat = cfg.r_threat_weight * delta_threat  # 塑形獎勵
        else:
            r_team_threat = 0.0
        st['prev_threat'] = threat_now

        # ===== 原有的 team reward =====
        nAliveEnd = int(np.sum(st['aliveAtt']))
        is_static = st.get('static_mode', False)

        if is_static:
            nKilledOrJammed = st['totalKills'] + st['totalJams']
            success_rate = nKilledOrJammed / max(1, st['Natt'])
        else:
            nFailed = st['penetrated'] + nAliveEnd
            success_rate = max(0.0, 1.0 - nFailed / st['Natt'])

        # 檢測重複目標
        n_duplicates = 0
        for i in range(st['Natt']):
            count = sum(1 for d in range(Ndef) if st['defs'][d].targetID == i)
            if count > 1:
                n_duplicates += (count - 1)

        Nnorm = max(1, st['Natt'])
        r_team = (cfg.r_step_time_pen +
                  cfg.r_jam_reward * (st['lastStepJams'] / Nnorm) +
                  cfg.r_penetration_pen * (st['lastStepPen'] / Nnorm) +
                  cfg.r_duplicate_penalty * (n_duplicates / Ndef) +
                  r_team_threat)  # ★ 加入威脅塑形

        if isDone:
            r_team += cfg.r_success_weight * success_rate
            if is_static and nAliveEnd == 0:
                r_team += cfg.r_static_clear_bonus

        rewards = r_indiv + r_team

        st['lastCandidates'] = self._build_candidates()
        obs_list = self._build_obs_all()

        info = {
            't': st['t'],
            'kills': st['totalKills'],
            'jams': st['totalJams'],
            'penetrated': st['penetrated'],
            'aliveEnd': nAliveEnd,
            'success_rate': success_rate,
            'is_static': is_static,
            'speeds': np.array([st['defs'][d].speed for d in range(Ndef)]),
            'threat_now': threat_now,  # ★ 診斷用:可看威脅值變化
            'n_unassigned': len(unassigned_alive),  # ★ 診斷用:未分配目標數
        }
        info['avg_speed'] = float(np.mean(info['speeds']))

        return obs_list, rewards, isDone, info

    # =================================================================
    # jammer  (對應 jammerStepFixed)
    # =================================================================
    def _jammer_step_fixed(self):
        cfg = self.cfg
        st = self.state
        if st['t'] < st['jammer_nextAvailableTime']:
            return
        if st['jammer_target'] == -1:
            best_dist = np.inf
            best_id = -1
            for i in range(st['Natt']):
                if st['aliveAtt'][i] and st['att'][i].state == 'UNKNOWN':
                    d = np.linalg.norm(st['att'][i].pos - cfg.jammer_pos)
                    if d <= cfg.jammer_range and d < best_dist:
                        best_dist = d
                        best_id = i
            st['jammer_target'] = best_id
        if st['jammer_target'] != -1:
            i = st['jammer_target']
            if not st['aliveAtt'][i] or st['att'][i].state != 'UNKNOWN':
                st['jammer_target'] = -1
                return
            success = (self.rng.random() < cfg.jammer_p_succ_const)
            if success:
                st['att'][i].state = 'JAMMED'
                st['aliveAtt'][i] = False
                st['lastStepJams'] += 1
                st['totalJams'] += 1
            else:
                st['att'][i].state = 'IMMUNE'
            st['jammer_nextAvailableTime'] = st['t'] + cfg.jammer_cooldown
            st['jammer_target'] = -1

    # =================================================================
    # candidates  (對應 buildCandidatesAllAgents)
    # =================================================================

    def _attacker_threat(self, i, st, cfg):
        """
        單一存活入侵者的威脅勢能 (0~1 歸一化)
        越靠近邊界威脅越高,用指數函數讓近距離時急遽上升。
        """
        if not st['aliveAtt'][i]:
            return 0.0
        p = st['att'][i].pos
        d = dist_point_to_rect_boundary(p, st['rect'])
        # exp(-d/scale) 使得 d=0 時勢能=1, d=scale 時勢能~0.37
        return float(np.exp(-d / cfg.threat_dist_scale))
    
    def _total_threat(self, st, cfg):
        """計算全場威脅總勢能"""
        return sum(self._attacker_threat(i, st, cfg) for i in range(st['Natt']))

    def _build_candidates(self):
        cfg = self.cfg
        st = self.state
        candidates = []
        for d in range(cfg.Ndef):
            valid_ids = []
            for i in range(st['Natt']):
                if (st['aliveAtt'][i] and st['att'][i].state == 'IMMUNE' and
                        st['aware'][d, i]):
                    valid_ids.append(i + 1)  # 1-based
            if valid_ids:
                t_rems = np.array([st['t_hit'][vid - 1] - st['t'] for vid in valid_ids])
                order = np.argsort(t_rems, kind='stable')
                valid_ids = [valid_ids[o] for o in order]
            cand = -np.ones(cfg.K_max, dtype=int)
            take = min(cfg.K_max, len(valid_ids))
            if take > 0:
                cand[:take] = valid_ids[:take]
            candidates.append(cand)
        return candidates

    # =================================================================
    # observation  (對應 buildObsForAgent)
    # =================================================================
    def _build_obs_all(self):
        return [self._build_obs_for_agent(d) for d in range(self.cfg.Ndef)]

    def _build_obs_for_agent(self, d):
        cfg = self.cfg
        st = self.state

        # self 特徵 (4)
        self_pos = st['defs'][d].pos / 200.0
        if st['defs'][d].targetID < 0:
            self_target_norm = -1.0
        else:
            self_target_norm = st['defs'][d].targetID / max(1, st['Natt'])
        self_mode = 1.0 if st['defs'][d].mode == 'PN_CHASE' else 0.0
        self_feat = np.array([self_pos[0], self_pos[1], self_target_norm, self_mode])

        # candidate 特徵 (K_max x 6 -> flatten)
        cand = st['lastCandidates'][d]
        if cand is None:
            cand = -np.ones(cfg.K_max, dtype=int)
        cand_feat = np.zeros((cfg.K_max, 6))
        for k in range(cfg.K_max):
            tid = cand[k]   # 1-based
            if tid <= 0:
                continue
            i = tid - 1
            p = st['att'][i].pos
            t_rem = max(st['t_hit'][i] - st['t'], 0.0)
            dist_self = np.linalg.norm(p - st['defs'][d].pos)
            is_targeted_by_others = 0
            for d2 in range(cfg.Ndef):
                if d2 == d:
                    continue
                if st['defs'][d2].targetID == tid:
                    is_targeted_by_others = 1
                    break
            cand_feat[k] = [
                p[0] / 200.0, p[1] / 200.0,
                min(t_rem / cfg.maxEpisodeSec, 1.0),
                min(dist_self / 200.0, 1.0),
                1.0 if st['aliveAtt'][i] else 0.0,  # ★ 修正:根據實際生死狀態
                is_targeted_by_others,
            ]
        cand_feat = cand_feat.reshape(-1)   # row-major: 對應 MATLAB reshape(cand_feat',[],1)

        # 隊友特徵 ((Ndef-1) x 4 -> flatten)
        mate_feat = np.zeros((cfg.Ndef - 1, 4))
        mi = 0
        for d2 in range(cfg.Ndef):
            if d2 == d:
                continue
            mate_pos = st['defs'][d2].pos / 200.0
            if st['defs'][d2].targetID < 0:
                mate_target_norm = -1.0
            else:
                mate_target_norm = st['defs'][d2].targetID / max(1, st['Natt'])
            mate_dist = np.linalg.norm(st['defs'][d2].pos - st['defs'][d].pos) / 200.0
            mate_feat[mi] = [mate_pos[0], mate_pos[1], mate_target_norm, min(mate_dist, 1.0)]
            mi += 1
        mate_feat = mate_feat.reshape(-1)

        # 全域特徵 (3)
        jammer_cd = max(st['jammer_nextAvailableTime'] - st['t'], 0.0)
        jammer_cd_norm = min(jammer_cd / max(cfg.jammer_cooldown, 1e-6), 1.0)
        nImm = sum(1 for i in range(st['Natt'])
                   if st['aliveAtt'][i] and st['att'][i].state == 'IMMUNE')
        nUnk = sum(1 for i in range(st['Natt'])
                   if st['aliveAtt'][i] and st['att'][i].state == 'UNKNOWN')
        nImm_norm = nImm / max(1, cfg.NattMax)
        nUnk_norm = nUnk / max(1, cfg.NattMax)
        global_feat = np.array([jammer_cd_norm, nImm_norm, nUnk_norm])

        obs = np.concatenate([self_feat, cand_feat, mate_feat, global_feat])
        obs[~np.isfinite(obs)] = 0.0
        obs = np.clip(obs, -1.0, 3.0)
        return obs


# 對應 appendAgentId
def append_agent_id(obs, agent_idx, Ndef):
    agent_id = np.zeros(Ndef)
    agent_id[agent_idx] = 1.0
    return np.concatenate([obs.reshape(-1), agent_id])


if __name__ == '__main__':
    # 自測：確認維度與基本行為
    env = MappoEnv(seed=0)
    env.configure(speed_range=[0.0, 0.0], static_mode=True)
    obs = env.reset()
    print('Ndef =', env.cfg.Ndef, ' Natt =', env.state['Natt'])
    print('每個 agent raw obs 維度 =', obs[0].shape, '(應為 49)')
    o_id = append_agent_id(obs[0], 0, env.cfg.Ndef)
    print('加 agent id 後 =', o_id.shape, '(應為 53)')
    # 跑幾步
    for step in range(3):
        ta = np.zeros(env.cfg.Ndef, dtype=int)   # 全選候選 0
        va = 4.0 * np.ones(env.cfg.Ndef)
        obs, r, done, info = env.step(ta, va)
        print(f'step {step}: reward={np.round(r,3)}, done={done}, '
              f'kills={info["kills"]}, succ={info["success_rate"]:.3f}')