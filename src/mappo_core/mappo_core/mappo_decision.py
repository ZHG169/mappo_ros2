"""
mappo_decision.py
封裝 actor 推論 + observation 轉接 + 動作→ID 對應。
提供跟 CBBA 一樣的輸出介面 (mu)，加上速度建議。

外部介面：
    mappo = MappoDecision(ckpt_path='...', origin_xy=(0,0), use_estimated=True)
    mu, def_speed_per_agent = mappo.decide(
        defs=..., tasks=..., aware=..., att_pos=..., est_pos=...,
        att_alive=..., att_state=..., t_now=..., jammer_next_avail=...,
        n_att_episode=...,
    )
"""

import numpy as np

from .mappo_inference import MappoInference
from .mappo_obs_adapter import RosObsBuilder, NDEF, K_MAX


class MappoDecision:
    def __init__(self, ckpt_path, origin_xy=(0.0, 0.0),
                 use_estimated=True, deterministic=True,
                 vMin=3.0, vMax=5.0, device='cpu'):
        self.infer = MappoInference(
            ckpt_path,
            deterministic=deterministic,
            vMin=vMin, vMax=vMax,
            device=device,
        )
        self.obs_builder = RosObsBuilder(
            origin_xy=origin_xy,
            use_estimated=use_estimated,
        )
        self.NDEF = NDEF
        self.K_MAX = K_MAX
        # 訓練動作: 0..K_MAX-1 = 候選索引, K_MAX = no-op
        self.NO_OP = K_MAX

    def decide(self, defs, tasks, aware, att_pos, est_pos,
               att_alive, att_state, t_now,
               jammer_next_avail, n_att_episode):
        """
        Returns
        -------
        mu : list[list[int]]    # 每台防守者要打的攻擊者 id list (0-based)
        def_speed : list[float] # 每台防守者的速度指令 (m/s)
        """
        # 1. 從 tasks 把 t_hit 抽成 array
        t_hit_arr = np.full(len(att_pos), np.inf)
        for tk in tasks:
            i = tk['id']
            if 0 <= i < len(att_pos):
                t_hit_arr[i] = tk['t_hit']

        # 2. 算每個 agent 的候選清單 (1-based id)
        cand_per_agent = self.obs_builder.build_candidates_only(
            defs=defs, est_pos=est_pos, att_pos=att_pos,
            att_alive=att_alive, att_state=att_state,
            aware=aware,
            t_hit=t_hit_arr,
            t_now=t_now,
        )

        # 3. 組 4 個 agent 的 53 維 obs
        obs_list = self.obs_builder.build_all(
            defs=defs, att_pos=att_pos, est_pos=est_pos,
            att_alive=att_alive, att_state=att_state,
            aware=aware,
            t_hit=t_hit_arr,
            t_now=t_now,
            jammer_next_avail=jammer_next_avail,
            n_att_episode=n_att_episode,
        )

        # 4. Actor forward (4 個 agent 一次跑)
        target_actions, velocities = self.infer.act_batch(obs_list)

        # 5. 動作 → 攻擊者 ID → mu
        mu = [[] for _ in range(self.NDEF)]
        for d in range(self.NDEF):
            a = int(target_actions[d])
            if a == self.NO_OP:
                # 維持原 target (跟訓練邏輯一致)
                old_tid = defs[d].get('targetID', -1)  # 1-based
                if old_tid > 0:
                    mu[d] = [old_tid - 1]  # 轉 0-based 給 CBBA 下游
                continue
            tid_1based = cand_per_agent[d][a] if a < self.K_MAX else -1
            if tid_1based > 0:
                mu[d] = [tid_1based - 1]  # 0-based,對齊 CBBA tasks['id']

        return mu, [float(v) for v in velocities]