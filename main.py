from numpy.typing import NDArray
from battle.envs import GridBattle
from battle.util import sample_map_1_sliding, sample_map_1
from battle.agents import AgentFunction


@AgentFunction
def myAgent(obs, action_space, obs_space) -> NDArray:
    return action_space.sample()


agent1 = myAgent()

env = GridBattle((agent1,) * 2, sample_map_1_sliding)  # This map changes over time!
# env = GridBattle((agent1,) * 2, sample_map_1)  # This map does not change over time!

env.run_game(200, render_mode="human")  # Change render_mode to None to disable rendering.