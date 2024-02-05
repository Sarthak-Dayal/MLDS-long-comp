import gym
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from battle.agents import Agent


class RLBattleAgent(Agent):

    def __init__(self, device):
        super().__init__()
        self.q_network = QNetwork((50, 50))
        self.target = QNetwork((50, 50))
        self.device = device
        self.q_network.to(device)
        self.target.to(device)

    def run_target_net(self, obs: Tensor):
        q_values = self.target(obs)
        return torch.argmax(q_values).cpu().numpy() + 1

    def run_q_net(self, obs: Tensor):
        q_values = self.q_network(obs)
        return torch.argmax(q_values).cpu().numpy() + 1

    def policy(self, obs: NDArray[int], action_space: gym.Space, obs_space: gym.Space) -> NDArray:
        obs = torch.from_numpy(obs).to(self.device).float()[None, :]
        return self.run_q_net(obs)


class QNetwork(nn.Module):
    def __init__(self, obs_space: tuple[int, ...]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.array(obs_space).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 25),  # 25 possibilities for the action
        )

    def forward(self, x):
        return self.network(x)
