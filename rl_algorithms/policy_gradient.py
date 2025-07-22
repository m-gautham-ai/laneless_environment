import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from lane_free_cav.envs.lane_free_cav_v0 import LaneFreeCAVEnv

# --- Policy Network ---
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        return dist

# --- REINFORCE Agent ---
class ReinforceAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99):
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.cpu().numpy().flatten()

    def finish_episode(self):
        # If no rewards were collected, do not perform an update
        if not self.rewards:
            return

        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

# --- Training Loop ---
def train():
    # Environment setup
    env = LaneFreeCAVEnv(render_mode="human", n_agents=1, road_width=30.0)
    state_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    print(state_dim)
    action_dim = env.action_spaces[env.possible_agents[0]].shape[0]
    print(action_dim)
    agent_id = env.possible_agents[0]

    agent = ReinforceAgent(state_dim, action_dim)

    for i_episode in range(1000):
        obs, info = env.reset()
        state = obs[agent_id]
        episode_reward = 0
        episode_length = 0
        termination_reason = "unknown"

        for t in range(500): # Max steps per episode
            action = agent.select_action(state)
            
            # The environment expects a dictionary of actions
            actions = {agent_id: action}
            obs, reward, terminated, truncated, info = env.step(actions)

            # If the agent terminated, it might not be in the next obs, but we still need its reward
            if agent_id in reward:
                agent.rewards.append(reward[agent_id])
                episode_reward += reward[agent_id]
                episode_length += 1

            # Check if agent is still alive for the next state
            if agent_id in env.agents:
                state = obs[agent_id]
            
            if terminated.get(agent_id) or truncated.get(agent_id):
                termination_reason = info.get(agent_id, {}).get('status', 'unknown')
                break

        agent.finish_episode()
        print(f"Episode {i_episode}\tLast reward: {episode_reward:.2f}\tEpisode length: {episode_length}\tReason: {termination_reason}")

    env.close()

if __name__ == '__main__':
    train()
