from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

DEVICE = "cuda"


class DummyEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.agents = [0, 1]
        self.observation_space = spaces.Dict({
            "feature1":
            spaces.Box(low=0, high=255, shape=(10, ), dtype=np.float32),
            "feature2":
            spaces.Box(low=0, high=255, shape=(1, 10, 10), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(2)

    def reset(self):
        obs, _, _, _ = self.step({i: 0 for i in self.agents})
        return obs

    def step(self, actions: Dict):
        assert isinstance(actions, Dict) and len(actions) == len(self.agents)
        obs, reward, done = {}, {}, {}
        for i in self.agents:
            obs[i] = {
                k: np.random.randn(*v.shape).astype(v.dtype)
                for k, v in self.observation_space.items()
            }
            reward[i] = np.random.random()
            done[i] = False
        return obs, reward, done, None

    def close(self) -> None:
        pass


class DummyNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.policy = nn.Linear(10 * 10 + 10, num_actions)
        self.baseline = nn.Linear(10 * 10 + 10, 1)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, input_dict, state=()):
        # [T, B, ...]
        f1, f2 = input_dict["feature1"], input_dict["feature2"]
        T, B, *_ = f1.shape
        f1 = torch.flatten(f1, 0, 1)
        f2 = torch.flatten(f2, 0, 1)
        f1 = F.relu(self.linear(f1))
        f2 = F.relu(self.conv(f2))
        f2 = torch.flatten(f2, start_dim=1)
        x = torch.cat([f1, f2], dim=-1)
        policy_logits = self.policy(x)
        baseline = self.baseline(x)
        action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                   num_samples=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (dict(policy_logits=policy_logits,
                     baseline=baseline,
                     action=action), tuple())