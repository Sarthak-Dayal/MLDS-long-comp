import gym
import numpy as np
from numpy.typing import NDArray

from battle import Agent
from battle.envs import GridBattle
from battle.util import sample_map_1_sliding, sample_map_1
from battle.agents import AgentFunction
from battle.agents.RLAgent import RLBattleAgent
import torch


class TestAgent(Agent):

    def __init__(self):
        super().__init__(obs_onehot=True)

    def policy(self, obs: NDArray[int], action_space: gym.Space, obs_space: gym.Space) -> NDArray:
        return 12 * np.ones(action_space.shape)


@AgentFunction
def Stationary(obs, action_space, obs_space) -> NDArray:
    # return action_space.sample()
    return 12 * np.ones(action_space.shape)


@AgentFunction
def Agent2(obs, action_space, obs_space) -> NDArray:
    # return action_space.sample()
    return 7 * np.ones(action_space.shape)


if __name__ == '__main__':
    agent1 = RLBattleAgent('cpu')
    agent1.q_network.load_state_dict(
        torch.load("runs/Reducing learning rate and train frequency__train__1__1708992769/train.cleanrl_model"))
    agent2 = TestAgent()

    env = GridBattle((agent1, agent2, TestAgent()), sample_map_1)  # This map changes over time!
    # env = GridBattle((agent1,) * 2, sample_map_1)  # This map does not change over time!

    env.run_game(200, render_mode="human")  # Change render_mode to None to disable rendering.
