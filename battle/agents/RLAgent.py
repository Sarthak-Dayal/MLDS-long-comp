import gym
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from battle import Agent


class RLBattleAgent(Agent):

    def __init__(self, device):
        super().__init__()
        self.net = QNetwork((50, 50))
        self.device = device
        self.net.to(device)

    def policy(self, obs: NDArray[int], action_space: gym.Space, obs_space: gym.Space) -> NDArray:
        obs = torch.from_numpy(obs).to(self.device)
        q_values = self.net(obs)
        return torch.argmax(q_values).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(self, obs_space: tuple[int, ...]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(obs_space).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 25),  # 25 possibilities for the action
        )

    def forward(self, x):
        return self.network(x)
