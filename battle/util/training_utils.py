import numpy as np
from numpy.typing import NDArray

def pad(grid: NDArray):
    arr = np.zeros((50, 50))
    arr[0:grid.shape[0], 0:grid.shape[1]] = grid
    return arr

# given array of numbers,
def get_action(grid, agent):
   friendly_coords = list(zip(*np.where(grid == 2)))
   action_out = np.zeros((50, 50))
   for coord in friendly_coords:
       grid[coord] = -1
       action_out[coord] = agent.policy(grid)
       grid[coord] = 2
   return action_out

def reward(grid):
    reward = (grid == 2).sum() - (grid > 2).sum()
    return reward