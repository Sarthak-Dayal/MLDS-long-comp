from typing import Sequence

import numpy as np
from gymnasium.spaces import MultiDiscrete
import torch

class Transition:

    def __init__(self, obs: MultiDiscrete, action: MultiDiscrete, reward: float, next_obs: MultiDiscrete, done: bool):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done


class ReplayBuffer:
    def __init__(self, size, device):
        self.buffer = []
        self.size = size
        self.next_idx = 0
        self.device = device

    def add(self, transition):
        if len(self.buffer) < self.size:
            self.buffer.append(transition)
        else:
            self.buffer[self.next_idx] = transition
        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size):
        indices = np.random.randint(len(self.buffer), size=batch_size)
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for idx in indices:
            transition = self.buffer[idx]
            obs.append(transition.obs)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_obs.append(transition.next_obs)
            dones.append(transition.done)

        return (torch.tensor(obs, dtype=torch.float32).to(self.device),
                torch.tensor(actions, dtype=torch.float32).to(self.device),
                torch.tensor(rewards, dtype=torch.float32).to(self.device),
                torch.tensor(next_obs, dtype=torch.float32).to(self.device),
                torch.tensor(dones, dtype=torch.bool).to(self.device))
