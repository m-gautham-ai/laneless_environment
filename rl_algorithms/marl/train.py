import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from lane_free_cav.envs.lane_free_cav_v0 import LaneFreeCAVEnv
from .mappo import Actor, Critic

def train():
    # --- Hyperparameters ---
    n_agents = 10
    episodes = 2000
    lr = 1e-4
    gamma = 0.99
    clip_epsilon = 0.2
    update_epochs = 10
    hidden_dim = 128
    save_interval = 100  # Save model every 100 episodes

    # --- Initialization ---
    env = LaneFreeCAVEnv(n_agents=n_agents, road_width=30.0, render_mode='human')
    agent_id_for_spaces = env.possible_agents[0]
    single_obs_dim = env.observation_spaces[agent_id_for_spaces].shape[0]
    single_action_dim = env.action_spaces[agent_id_for_spaces].shape[0]
    global_state_dim = single_obs_dim * n_agents

    actor = Actor(single_obs_dim, single_action_dim, hidden_dim)
    critic = Critic(global_state_dim, hidden_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    # Create a directory to save models
    if not os.path.exists('models'):
        os.makedirs('models')

    print(f"Starting MAPPO training with {n_agents} agents.")

    for episode in range(episodes):
        obs, _ = env.reset()
        done = {a: False for a in env.possible_agents}
        episode_reward = 0
        buffer = []

        while not all(done.values()):
            # --- Collect actions for all agents ---
            actions = {}
            log_probs = {}
            for agent_id in env.agents:
                if agent_id not in obs:
                    continue
                agent_obs = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                action_mean = actor(agent_obs)
                dist = torch.distributions.Normal(action_mean, 0.5)  # Using a fixed std dev for simplicity
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                
                actions[agent_id] = torch.clamp(action, -1.0, 1.0).squeeze().numpy()
                log_probs[agent_id] = log_prob

            # --- Step the environment ---
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # --- Store experience in a buffer ---
            # Ensure there are observations to concatenate
            if not obs:
                break
            padded_obs = [obs.get(agent_id, np.zeros(single_obs_dim)) for agent_id in env.possible_agents]
            global_state = torch.FloatTensor(np.concatenate(padded_obs)).unsqueeze(0)
            buffer.append((obs, global_state, actions, log_probs, rewards, done))

            obs = next_obs
            done = {a: terminated.get(a, False) or truncated.get(a, False) for a in env.possible_agents}
            
            # Simple reward aggregation for logging
            episode_reward += sum(rewards.values())

        # --- Perform PPO update ---
        # --- Perform PPO update ---
        # --- Calculate shared returns ---
        num_steps = len(buffer)
        returns = torch.zeros(num_steps)
        R = 0
        for i in reversed(range(num_steps)):
            rewards_b = buffer[i][4]
            done_b = buffer[i][5]

            # If all agents are done, reset the return
            if all(done_b.values()):
                R = 0
            
            # Sum rewards for a shared return
            current_reward = sum(rewards_b.values())
            R = current_reward + gamma * R
            returns[i] = float(R)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # --- Batching data from buffer ---
        batch_obs, batch_actions, batch_log_probs = [], [], []
        for step in buffer:
            obs_b, _, actions_b, log_probs_b, _, _ = step
            for agent_id in env.possible_agents:
                if agent_id in obs_b and agent_id in actions_b:
                    batch_obs.append(torch.FloatTensor(obs_b[agent_id]).unsqueeze(0))
                    batch_actions.append(torch.from_numpy(actions_b[agent_id]))
                    batch_log_probs.append(log_probs_b[agent_id].unsqueeze(0))

        old_obs_b = torch.cat(batch_obs)
        old_actions_b = torch.stack(batch_actions).float()
        old_log_probs_b = torch.cat(batch_log_probs).detach()
        global_states_b = torch.cat([step[1] for step in buffer])

        with torch.no_grad():
            state_values = critic(global_states_b).squeeze()
        advantages = (returns - state_values).detach()
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(update_epochs):
            # --- Actor Loss ---
            action_mean = actor(old_obs_b)
            dist = torch.distributions.Normal(action_mean, 0.5)
            new_log_probs = dist.log_prob(old_actions_b).sum(dim=1)
            ratio = torch.exp(new_log_probs - old_log_probs_b)

            # Align advantages for actor loss calculation
            aligned_advantages = normalized_advantages.repeat_interleave(n_agents)
            if aligned_advantages.shape[0] != ratio.shape[0]:
                min_len = min(aligned_advantages.shape[0], ratio.shape[0])
                aligned_advantages = aligned_advantages[:min_len]
                ratio = ratio[:min_len]

            surr1 = ratio * aligned_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * aligned_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # --- Critic Loss ---
            state_values = critic(global_states_b)
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # --- Update Networks ---
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            total_loss = actor_loss + critic_loss
            total_loss.backward()

            actor_optimizer.step()
            critic_optimizer.step()

        print(f"Episode {episode}: Total Reward: {episode_reward:.2f}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

        # --- Save the model periodically ---
        if episode % save_interval == 0:
            torch.save(actor.state_dict(), f'models/actor_episode_{episode}.pth')
            torch.save(critic.state_dict(), f'models/critic_episode_{episode}.pth')
            print(f"Models saved at episode {episode}")

    env.close()

if __name__ == '__main__':
    train()
