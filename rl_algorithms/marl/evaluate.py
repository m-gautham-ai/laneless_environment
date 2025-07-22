import torch
import numpy as np
import argparse
from lane_free_cav.envs.lane_free_cav_v0 import LaneFreeCAVEnv
from .mappo import Actor

def evaluate(model_path):
    # --- Hyperparameters ---
    n_agents = 10
    hidden_dim = 128

    # --- Initialization ---
    env = LaneFreeCAVEnv(n_agents=n_agents, road_width=30.0, render_mode='human')
    agent_id_for_spaces = env.possible_agents[0]
    single_obs_dim = env.observation_spaces[agent_id_for_spaces].shape[0]
    single_action_dim = env.action_spaces[agent_id_for_spaces].shape[0]

    # --- Load Actor Model ---
    actor = Actor(single_obs_dim, single_action_dim, hidden_dim)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()  # Set the actor to evaluation mode

    print(f"Evaluating model: {model_path}")

    obs, _ = env.reset()
    done = {a: False for a in env.possible_agents}
    episode_reward = 0

    while not all(done.values()):
        actions = {}
        with torch.no_grad():
            for agent_id in env.agents:
                if agent_id not in obs:
                    continue
                agent_obs = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                action_mean = actor(agent_obs)
                actions[agent_id] = torch.clamp(action_mean, -1.0, 1.0).squeeze().numpy()

        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        obs = next_obs
        done = {a: terminated.get(a, False) or truncated.get(a, False) for a in env.possible_agents}
        episode_reward += sum(rewards.values())

    print(f"Evaluation finished. Total Reward: {episode_reward:.2f}")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved actor model')
    args = parser.parse_args()
    evaluate(args.model_path)
