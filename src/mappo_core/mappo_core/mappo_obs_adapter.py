"""
mappo_obs_adapter.py
ROS 端的 observation 轉接層：把 ROS 即時資料組成跟訓練時 _build_obs_for_agent
完全一致的 53 維 observation。
"""

import numpy as np


# 現有 checkpoint 的訓練預設值（需與使用中的模型結構相容）
NDEF = 4
K_MAX = 5
N_ACTIONS = K_MAX + 1
NATT_MAX = 12
SENSOR_RANGE = 75.0
MAX_EPISODE_SEC = 200.0
JAMMER_COOLDOWN = 1.0
NORM_COORD = 200.0
CX, CY = 100.0, 80.0


def build_candidates_for_agent(d, def_pos_2d, attackers, t_now):
    valid = [a for a in attackers
             if a['alive'] and a['state'] == 'IMMUNE' and a['aware_by_d']]
    valid.sort(key=lambda a: a['t_hit'] - t_now)
    cand = [-1] * K_MAX
    for k in range(min(K_MAX, len(valid))):
        cand[k] = valid[k]['id'] + 1
    return cand


def build_obs_for_agent(d, defenders, attackers, candidates_d,
                        t_now, jammer_next_avail, n_att_episode):
    me = defenders[d]

    # self 特徵 (4)
    self_pos = me['pos'] / NORM_COORD
    if me['targetID'] < 0:
        self_target_norm = -1.0
    else:
        self_target_norm = me['targetID'] / max(1, n_att_episode)
    self_mode = 1.0 if me['mode'] == 'PN_CHASE' else 0.0
    self_feat = np.array([self_pos[0], self_pos[1],
                          self_target_norm, self_mode])

    # 候選特徵 (K_MAX x 6)
    cand_feat = np.zeros((K_MAX, 6))
    for k in range(K_MAX):
        tid = candidates_d[k]
        if tid <= 0:
            continue
        i = tid - 1
        a = attackers[i]
        p = a['pos']
        t_rem = max(a['t_hit'] - t_now, 0.0)
        dist_self = np.linalg.norm(p - me['pos'])
        is_targeted_by_others = 0
        for d2 in range(NDEF):
            if d2 == d:
                continue
            if defenders[d2]['targetID'] == tid:
                is_targeted_by_others = 1
                break
        cand_feat[k] = [
            p[0] / NORM_COORD,
            p[1] / NORM_COORD,
            min(t_rem / MAX_EPISODE_SEC, 1.0),
            min(dist_self / NORM_COORD, 1.0),
            1.0,
            is_targeted_by_others,
        ]
    cand_feat = cand_feat.reshape(-1)

    # 隊友特徵 ((Ndef-1) x 4)
    mate_feat = np.zeros((NDEF - 1, 4))
    mi = 0
    for d2 in range(NDEF):
        if d2 == d:
            continue
        mate = defenders[d2]
        mp = mate['pos'] / NORM_COORD
        if mate['targetID'] < 0:
            mate_target_norm = -1.0
        else:
            mate_target_norm = mate['targetID'] / max(1, n_att_episode)
        mate_dist = np.linalg.norm(mate['pos'] - me['pos']) / NORM_COORD
        mate_feat[mi] = [mp[0], mp[1], mate_target_norm,
                         min(mate_dist, 1.0)]
        mi += 1
    mate_feat = mate_feat.reshape(-1)

    # 全域特徵 (3)
    jammer_cd = max(jammer_next_avail - t_now, 0.0)
    jammer_cd_norm = min(jammer_cd / max(JAMMER_COOLDOWN, 1e-6), 1.0)
    nImm = sum(1 for a in attackers
               if a['alive'] and a['state'] == 'IMMUNE')
    nUnk = sum(1 for a in attackers
               if a['alive'] and a['state'] == 'UNKNOWN')
    global_feat = np.array([jammer_cd_norm,
                            nImm / max(1, NATT_MAX),
                            nUnk / max(1, NATT_MAX)])

    obs = np.concatenate([self_feat, cand_feat, mate_feat, global_feat])
    obs[~np.isfinite(obs)] = 0.0
    obs = np.clip(obs, -1.0, 3.0)
    return obs


def append_agent_id(obs_49, d):
    agent_id = np.zeros(NDEF)
    agent_id[d] = 1.0
    return np.concatenate([obs_49, agent_id])


class RosObsBuilder:
    """把 CBBA 程式裡的 defs / tasks / aware / t_hit 直接餵進來，組 53 維 obs。"""

    def __init__(self, origin_xy=(0.0, 0.0), use_estimated=True):
        """
        origin_xy : (float, float)
            「保護區中心」在 ROS 座標系裡的 (x, y)。
            訓練時保護區中心固定在 (100, 80)，本層做:
                pos_train = (pos_ros - origin_xy) + (CX, CY)
            ROS 端保護區中心就在 (100, 80) 時 origin_xy 給 (0, 0)。
        use_estimated : bool
            True = UKF 估測 (est_pos)；False = 真值 (att_pos, 上帝視角)。
        """
        self.origin = np.array(origin_xy, dtype=float)
        self.use_estimated = use_estimated

    def _to_train_xy(self, pos_ros):
        p = np.asarray(pos_ros)[:2] - self.origin
        return p + np.array([CX, CY])

    def build_all(self, defs, att_pos, est_pos,
                  att_alive, att_state, aware,
                  t_hit, t_now, jammer_next_avail, n_att_episode):
        Natt = len(att_pos)
        src_pos = est_pos if self.use_estimated else att_pos

        defenders = []
        for d in range(NDEF):
            defenders.append({
                'pos': self._to_train_xy(defs[d]['pos']),
                'mode': defs[d].get('mode', 'PATROL'),
                'targetID': defs[d].get('targetID', -1),
            })

        attackers = []
        for i in range(Natt):
            attackers.append({
                'id': i,
                'pos': self._to_train_xy(src_pos[i]),
                't_hit': float(t_hit[i]) if t_hit[i] is not None else float('inf'),
                'state': att_state[i],
                'alive': bool(att_alive[i]),
                'aware_by_d': False,
            })

        obs_list = []
        for d in range(NDEF):
            for i in range(Natt):
                attackers[i]['aware_by_d'] = bool(aware[d, i])
            cand = build_candidates_for_agent(d, defenders[d]['pos'],
                                              attackers, t_now)
            obs49 = build_obs_for_agent(d, defenders, attackers, cand,
                                        t_now, jammer_next_avail,
                                        n_att_episode)
            obs53 = append_agent_id(obs49, d)
            obs_list.append(obs53)

        return obs_list

    def build_candidates_only(self, defs, est_pos, att_pos,
                              att_alive, att_state, aware, t_hit, t_now):
        Natt = len(att_pos)
        src_pos = est_pos if self.use_estimated else att_pos
        attackers = []
        for i in range(Natt):
            attackers.append({
                'id': i,
                'pos': self._to_train_xy(src_pos[i]),
                't_hit': float(t_hit[i]) if t_hit[i] is not None else float('inf'),
                'state': att_state[i],
                'alive': bool(att_alive[i]),
                'aware_by_d': False,
            })
        out = []
        for d in range(NDEF):
            def_pos = self._to_train_xy(defs[d]['pos'])
            for i in range(Natt):
                attackers[i]['aware_by_d'] = bool(aware[d, i])
            out.append(build_candidates_for_agent(d, def_pos, attackers, t_now))
        return out