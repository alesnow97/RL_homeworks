import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# =========================
# Neural Network Modules
# =========================

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), activation()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim, 2*action_dim, hidden_sizes)
        self.log_std_min, self.log_std_max = -20, 2

    def forward(self, state):
        mu_logstd = self.net(state)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (log_std + 1.0) * (self.log_std_max - self.log_std_min)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(-1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(-1)
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# Soft Actor-Critic Agent
# =========================

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Policy
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        # Critics
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)
        self.q1_target = MLP(state_dim + action_dim, 1)
        self.q2_target = MLP(state_dim + action_dim, 1)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if eval:
            mu, _ = self.policy.forward(state)
            action = torch.tanh(mu)
        else:
            action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Sample next action
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_actions], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], dim=-1))
            q_target = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * q_target

        # Update Q-functions
        q1_val = self.q1(torch.cat([states, actions], dim=-1))
        q2_val = self.q2(torch.cat([states, actions], dim=-1))
        q1_loss = nn.MSELoss()(q1_val, target_value)
        q2_loss = nn.MSELoss()(q2_val, target_value)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Update Policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.q2(torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update targets
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# =========================
# Training Loop
# =========================

def train_sac(env_name="Pendulum-v1", episodes=200):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer()
    batch_size = 256
    start_steps = 10000

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            if len(replay_buffer) < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

        print(f"Episode {ep}: Reward = {episode_reward:.2f}")

    env.close()
    return agent


if __name__ == "__main__":
    trained_agent = train_sac()
