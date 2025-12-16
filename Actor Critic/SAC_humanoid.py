# Soft Actor-Critic (SAC)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import copy
import numpy as np
import gymnasium as gym
from collections import deque
import random
import time
import os
import sys
import pickle

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim) 
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mean = self.mean(x) 
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # numerical stability to prevent vanishing and exploding variance
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t) # Enforce action bounds from -1 to 1, but also changes probability distribution
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6) # Jacobian correction since we changed probability distribution by squishing it using tanh
        # π(a|s) = µ(u|s) |det da/du|^−1   Change of variables formula
        # da /du = diag(1−tanh^2(u))
        # log π(a|s) = log µ(u|s) − sum_i[log(1−tanh^2(u_i))]
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
        
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
def train(checkpoint_path):
    # Training and test configs
    epochs = 2000
    test_epochs_freq = 100
    steps_per_epoch = 256
    batch_size = 256
    num_test_episodes = 10
    num_trajectories = 10
    best_test_reward = 0

    # Reproducibility
    train_seed = 42
    test_seed = 55
    set_seed = False if "--no-seed" in sys.argv else True
    if set_seed and train_seed is not None:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        random.seed(train_seed)

    # Environment configs
    # Environment configs
    env_name = "Humanoid-v5"
    max_episode_steps = 1000
    
    # Agent configs
    gamma = 0.99
    policy_lr = 3e-4
    q_lr = 3e-4
    alpha_lr = 3e-4
    
    # Replay buffer configs
    replay_buffer_size = 100000
    
    # Target network configs
    tau = 0.005


    # Initialize the environment
    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    obs_normalizer = RunningMeanStd(shape=(obs_dim,))
    
    # Initialize the actor and critic networks
    policy = PolicyNetwork(obs_dim, action_dim)
    q1 = QNetwork(obs_dim, action_dim)
    q2 = QNetwork(obs_dim, action_dim)
    
    # Initialize the target networks and set parameters equal to the original networks
    # policy_target = copy.deepcopy(policy)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    
    # Initialize the optimizers
    policy_optim = optim.Adam(policy.parameters(), lr=policy_lr)
    q1_optim = optim.Adam(q1.parameters(), lr=q_lr)
    q2_optim = optim.Adam(q2.parameters(), lr=q_lr)

    # Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(torch.device("cpu"))).item()
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = optim.Adam([log_alpha], lr=alpha_lr)
    alpha = log_alpha.exp()
    
    # Initialize the replay buffer
    replay_buffer = deque(maxlen=replay_buffer_size)
    
    # Training loop
    for epoch in range(epochs):
        # Collect a batch of trajectories
        for i in range(num_trajectories):
            seed = train_seed + epoch if set_seed and train_seed is not None else None
            obs, info = env.reset(seed=seed)
            obs = obs_normalizer.normalize(obs)
            done = False
            while not done:
                # Sample an action from the policy
                with torch.no_grad():
                    action, _ = policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = action.squeeze(0).numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Update normalizer with raw next_obs, then normalize it
                obs_normalizer.update(np.array([next_obs]))
                next_obs = obs_normalizer.normalize(next_obs)
                
                done = terminated or truncated
                replay_buffer.append((obs, action, reward, next_obs, done))
                obs = next_obs
        
        # Sample a batch of steps from the replay buffer
        for i in range(steps_per_epoch):
            if len(replay_buffer) < batch_size:
                break
            batch = random.sample(replay_buffer, batch_size)
            obs, action, reward, next_obs, done = zip(*batch)
            
            # Convert to tensors
            obs = torch.tensor(np.array(obs), dtype=torch.float32)
            action = torch.tensor(np.array(action), dtype=torch.float32)
            reward = torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1)
            next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
            done = torch.tensor(np.array(done), dtype=torch.int8).unsqueeze(1)
            
            # Compute targets for Q-functions
            with torch.no_grad():
                next_action, next_log_prob = policy.sample(next_obs)  # a' = π(s')
                next_state_action = torch.cat([next_obs, next_action], dim=1)
                q1_next = q1_target(next_state_action)  # q1(s',a') - using target q networks for stability
                q2_next = q2_target(next_state_action)  # q2(s',a')
                min_q_next = torch.min(q1_next, q2_next)  # Q(s',a')
                y = reward + gamma * (1 - done) * (min_q_next - alpha * next_log_prob)
                # y = r(s,a) + γ * (Qθ(s',a') - α * log(π(a'|s')))

            # Update Q-functions
            state_action = torch.cat([obs, action], dim=1)
            q1_loss = F.mse_loss(q1(state_action), y)  # J(θ) = E(s,a) ~ D [1/2 * (Qθ(s,a) - y)^2]
            q2_loss = F.mse_loss(q2(state_action), y)
            
            q1_optim.zero_grad()
            q1_loss.backward()
            q1_optim.step()

            q2_optim.zero_grad()
            q2_loss.backward()
            q2_optim.step()

            # Update Policy
            new_action, log_prob = policy.sample(obs)  # a' = π(s')
            q1_new = q1(torch.cat([obs, new_action], dim=1))  # q1(s',a')
            q2_new = q2(torch.cat([obs, new_action], dim=1))  # q2(s',a')
            min_q_new = torch.min(q1_new, q2_new)  # Q(s',a')
            
            policy_loss = -(min_q_new - alpha * log_prob).mean()    
            # -J(πφ) = E s ~ D, ϵ ~ N [Qθ(s,a) - α * log(πφ(a|s))]
            # where a = fφ(ϵ;s) is the action sampled using the reparameterization trick

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # Update Alpha
            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            alpha = log_alpha.exp()

            # Soft update target networks
            for target_param, param in zip(q1_target.parameters(), q1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
            for target_param, param in zip(q2_target.parameters(), q2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        # Test the policy
        if (epoch + 1) % test_epochs_freq == 0:
            test_rewards = []
            for i in range(num_test_episodes):
                obs, info = env.reset(seed = test_seed + i if set_seed and test_seed is not None else None)
                obs = obs_normalizer.normalize(obs)
                done = False
                episode_reward = 0
                while not done:
                    with torch.no_grad():
                        action, _ = policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action = action.squeeze(0).numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    obs = obs_normalizer.normalize(obs)
                    done = terminated or truncated
                    episode_reward += reward
                if episode_reward > best_test_reward:
                    best_test_reward = episode_reward
                    torch.save(policy.state_dict(), checkpoint_path)
                    with open("checkpoints/sac_humanoid_obs_normalizer.pkl", "wb") as f:
                        pickle.dump(obs_normalizer, f)
                        

                test_rewards.append(episode_reward)
            print(f"Epoch {epoch+1}, Avg Test Reward: {np.mean(test_rewards)}")

    env.close()

# Demo the policy
def demo(policy, obs_normalizer, num_times):
    env = gym.make("Humanoid-v5", render_mode="human")
    for i in range(num_times):    
        obs, info = env.reset()
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)
        done = False
        tot_reward = 0
        policy.eval()
        with torch.no_grad():
            while not done:
                action, _ = policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = action.squeeze(0).numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                if obs_normalizer is not None:
                    obs = obs_normalizer.normalize(obs)
                tot_reward += reward
                done = terminated or truncated
            print(f"Total reward for episode {i+1}: {tot_reward}")

if __name__ == "__main__":
    train_flag = True
    render_flag = False
    checkpoint_path = "checkpoints/sac_humanoid_policy.pth"
    if train_flag:
        start_time = time.time()
        train(checkpoint_path)
        end_time = time.time()
        print(f"Total Training Time: {(end_time - start_time)/60:.2f} minutes")
    elif render_flag:
        env = gym.make("Humanoid-v5", render_mode="human")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        
        policy = PolicyNetwork(obs_dim, action_dim)
        policy.load_state_dict(torch.load(checkpoint_path))
        
        obs_normalizer = None
        if os.path.exists("checkpoints/sac_humanoid_obs_normalizer.pkl"):
            with open("checkpoints/sac_humanoid_obs_normalizer.pkl", "rb") as f:
                obs_normalizer = pickle.load(f)
                
        demo(policy, obs_normalizer, 5)
        