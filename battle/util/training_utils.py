import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


def pad_onehot(grid: NDArray):
    # Calculate padding for height and width
    pad_height = max(0, 50 - grid.shape[1])
    pad_width = max(0, 50 - grid.shape[2])

    # Apply padding to the array
    padded_array = np.pad(grid, ((0, 0), (0, pad_height), (0, pad_width)))

    return padded_array


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


def action_to_batch_actions(action: NDArray, coords):
    return action[tuple(zip(*coords))]


def obs_to_batch_grids(grid: Tensor):
    friendly_coords = list(zip(*np.where(grid == 2)))

    grids = []
    for coord in friendly_coords:
        grid[coord] = -1
        grids.append(np.copy(grid))
        grid[coord] = 2

    return torch.tensor(grids), friendly_coords


def reward(grid):
    return -(grid > 2).sum()
