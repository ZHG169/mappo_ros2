"""
mappo_inference.py
ROS 端的 MAPPO actor 推論模組。
只包含載入訓練好的 .pt 權重後做推論的最小邏輯，不依賴環境/訓練程式。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadActor(nn.Module):
    """必須跟訓練時的結構完全一致，否則載入權重會失敗。"""
    def __init__(self, obs_dim=53, n_actions=6):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head_logits = nn.Linear(64, n_actions)
        self.head_v_mean = nn.Linear(64, 1)
        self.head_v_log_std = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.head_logits(h), self.head_v_mean(h), self.head_v_log_std(h)


class MappoInference:
    """包裝 actor，提供給 ROS 端用的簡單介面。"""

    def __init__(self, ckpt_path, deterministic=True,
                 obs_dim=53, n_actions=6, vMin=3.0, vMax=5.0,
                 device='cpu'):
        self.device = torch.device(device)
        self.actor = DualHeadActor(obs_dim, n_actions).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if 'actor' in ckpt:
            self.actor.load_state_dict(ckpt['actor'])
        else:
            self.actor.load_state_dict(ckpt)
        self.actor.eval()

        self.deterministic = deterministic
        self.vMid = (vMin + vMax) / 2
        self.vRange = (vMax - vMin) / 2
        self.vMin = vMin
        self.vMax = vMax
        self.n_actions = n_actions

    @torch.no_grad()
    def act(self, obs_53):
        obs = torch.as_tensor(obs_53, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        logits, v_mean, v_log_std = self.actor(obs)
        logits = logits.squeeze(0).cpu().numpy()
        v_mean = float(v_mean.item())

        if self.deterministic:
            target_action = int(np.argmax(logits))
            velocity = self.vMid + self.vRange * np.tanh(v_mean)
        else:
            m = logits.max()
            ex = np.exp(logits - m)
            probs = ex / ex.sum()
            target_action = int(np.random.choice(self.n_actions, p=probs))
            mean_scaled = self.vMid + self.vRange * np.tanh(v_mean)
            std = np.exp(float(v_log_std.item()))
            std = float(np.clip(std, 0.1, 1.0))
            velocity = mean_scaled + std * np.random.randn()

        velocity = float(np.clip(velocity, self.vMin, self.vMax))
        return target_action, velocity

    @torch.no_grad()
    def act_batch(self, obs_list):
        obs_batch = torch.as_tensor(np.stack(obs_list), dtype=torch.float32,
                                    device=self.device)
        logits, v_mean, _ = self.actor(obs_batch)
        logits = logits.cpu().numpy()
        v_mean = v_mean.squeeze(-1).cpu().numpy()

        if self.deterministic:
            target_actions = np.argmax(logits, axis=1).astype(int)
        else:
            m = logits.max(axis=1, keepdims=True)
            ex = np.exp(logits - m)
            probs = ex / ex.sum(axis=1, keepdims=True)
            target_actions = np.array([
                np.random.choice(self.n_actions, p=p) for p in probs
            ], dtype=int)

        velocities = self.vMid + self.vRange * np.tanh(v_mean)
        velocities = np.clip(velocities, self.vMin, self.vMax)
        return target_actions, velocities