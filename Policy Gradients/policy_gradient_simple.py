import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy
class Policy(Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(8, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utility functions
def compute_rewards_to_go(rewards, gamma=0.99):
    T = len(rewards) 
    gamma_vector = np.array([gamma**t for t in range(T)])
    reversed_gamma_vector = np.flip(gamma_vector)
    rewards_to_go = np.convolve(rewards, reversed_gamma_vector)[-T:]
    return rewards_to_go

# Train
def train():
    # Variables
    num_epochs = 1000
    num_episodes_per_epoch = 32
    test_every = 100 # tests every n epochs
    seed = 42

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize
    policy = Policy().to(device)
    env = gym.make("LunarLander-v3")
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Train
    for epoch in range(num_epochs):
        policy.train()
        observations, actions, rewards, advantages = [], [], [], []
        
        # Collect data - disable autograd
        with torch.no_grad():
            for episode in range(num_episodes_per_epoch):
                obs, info = env.reset(seed=seed + epoch * num_episodes_per_epoch + episode)
                done = False
            
                while not done:
                    logits = policy(torch.tensor(obs, device=device, dtype=torch.float32))
                    action = Categorical(logits=logits).sample().item()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    obs = next_obs
                    done = terminated or truncated

            # Compute rewards to go
            rewards_to_go = compute_rewards_to_go(rewards, gamma=0.99)
            advantages.extend(rewards_to_go)
            
        # Convert to tensors and move to GPU
        observations = torch.tensor(np.array(observations), device=device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=device, dtype=torch.int64)
        logits = policy(observations)
        act_dist = Categorical(logits=logits)
        log_prob = act_dist.log_prob(actions)
        entropy = act_dist.entropy()
        
        # Subtract baseline 
        advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        
        # Compute loss taking the mean over all steps instead of all episodes
        loss = -(log_prob * advantages).mean() - 0.01 * entropy.mean()

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

        # Test policy after 100 episodes
        if (epoch+1) % test_every == 0:
            test_rewards = []   
            policy.eval() 
            with torch.no_grad():      
                for i in range(5):
                    obs, info = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        logits = policy(torch.tensor(obs, device=device, dtype=torch.float32))
                        action = torch.argmax(logits).item()
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                    test_rewards.append(episode_reward)
            print(f"Epoch {epoch + 1} - Average reward: {np.mean(test_rewards)}")

if __name__ == "__main__":
    train()