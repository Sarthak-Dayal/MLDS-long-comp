# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from battle import GridBattle, RandomAgent, RLBattleAgent, pad
from battle.util import sample_map_1, sample_map_1_sliding, get_action, ReplayBuffer, Transition, obs_to_batch_grids, action_to_batch_actions

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MLDS Comp"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "GridBattle"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 100
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run = None
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        run.log_code()

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    agent1 = RLBattleAgent(device)
    agent2 = RandomAgent()
    # env setup
    env = GridBattle((agent1, agent2), sample_map_1)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    optimizer = optim.Adam(agent1.q_network.parameters(), lr=args.learning_rate)
    agent1.target.load_state_dict(agent1.q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        device
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset(seed=args.seed)
    action_space = obs[0].shape
    obs[0] = torch.from_numpy(pad(obs[0])).float()
    obs[1] = torch.from_numpy(pad(obs[1])).float()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        grid_batch, coords = obs_to_batch_grids(obs[0])
        if random.random() < epsilon:
            action = env.get_action_space(agent1).sample()
        else:
            action = agent1.policy(obs[0], env.get_action_space(agent1), None)

        action_batch = action_to_batch_actions(action, coords)

        random_action = agent2.policy(obs, env.action_spaces[1], env.observation_spaces[1])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step([action, random_action])
        next_obs[0] = torch.from_numpy(pad(next_obs[0])).float()
        next_obs[1] = torch.from_numpy(pad(next_obs[1])).float()

        next_obs_batch, new_coords = obs_to_batch_grids(next_obs[0])

        # Convert coords and new_coords to sets
        coords_set = set(map(tuple, coords))
        new_coords_set = set(map(tuple, new_coords))

        # Find the common coordinates
        common_coords = coords_set & new_coords_set

        # Convert common_coords back to a list of tuples
        common_coords = list(map(list, common_coords))

        # Find the indices of the common coordinates in coords and new_coords
        indices_in_coords = [coords.index(tuple(coord)) for coord in common_coords]
        indices_in_new_coords = [new_coords.index(tuple(coord)) for coord in common_coords]

        # Use these indices to filter grid_batch and next_obs_batch
        filtered_grid_batch = grid_batch[indices_in_coords]
        filtered_next_obs_batch = next_obs_batch[indices_in_new_coords]

        # print(f"reward: {reward}")
        writer.add_scalar("charts/step_reward", reward, global_step)
        # print(f"self_agents: {(obs[0] == 2).sum()}")
        writer.add_scalar("charts/self_agents", (obs[0] == 2).sum(), global_step)
        # print(f"enemy_agents: {(obs[0] > 2).sum()}")
        writer.add_scalar("charts/enemy_agents", (obs[0] > 2).sum(), global_step)


        # TRY NOT TO MODIFY: record rewards for plotting purposes        
        if info and "episode" in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        rb.add_batch(Transition.build_batch(filtered_grid_batch, action_batch, float(reward), filtered_next_obs_batch, termination or truncation))

        if termination or truncation:
            obs, info = env.reset()
            obs[0] = torch.from_numpy(pad(obs[0])).float()
            obs[1] = torch.from_numpy(pad(obs[1])).float()
        else:
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                obs_batch, actions, rewards, next_obs_batch, dones = rb.sample(args.batch_size)
                with torch.no_grad():
                    # target_max, _ = agent1.run_target_net(next_obs).max(dim=1)
                    target_max = agent1.run_target_net(next_obs_batch)
                    td_target = rewards.flatten() + args.gamma * target_max * (1 - dones.int().flatten())
                # obs_agent1, obs_agent2 = torch.chunk(obs, chunks=2, dim=1)
                # obs_agent1 = obs_agent1.squeeze(1)
                # obs_agent2 = obs_agent2.squeeze(1)
                old_val = agent1.q_network(obs_batch).gather(1, actions.long()[None, :]).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(agent1.target.parameters(), agent1.q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent1.q_network.state_dict(), model_path)

        if run:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            run.log_artifact(artifact)
        print(f"model saved to {model_path}")


    #     from cleanrl.cleanrl_utils.evals.dqn_eval import evaluate
    #
    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=QNetwork,
    #         device=device,
    #         epsilon=0.05,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)
    #
    #     if args.upload_model:
    #         from cleanrl.cleanrl_utils.huggingface import push_to_hub
    #
    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    env.close()
    writer.close()
