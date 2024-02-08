import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from torch import Tensor


def pad(grid: NDArray):
    arr = np.zeros((50, 50))
    arr[0:grid.shape[0], 0:grid.shape[1]] = grid
    return arr


# given array of numbers,
def get_action(grid: Tensor, agent, action_space: tuple):

    friendly_coords = list(zip(*np.where(grid == 2)))
    action_out = np.zeros((50, 50))
    for coord in friendly_coords:
        grid[coord] = -1
        action_out[coord] = agent.policy(grid.numpy(), None, None)
        grid[coord] = 2

    return action_out[0:action_space[0], 0:action_space[1]]


def reward(grid):
    return (grid == 2).sum() - (grid > 2).sum()
