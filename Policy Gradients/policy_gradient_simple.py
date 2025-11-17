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
    num_epochs = 2000
    num_episodes_per_epoch = 64
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
        multi_episode_log_probs, multi_episode_rewards_to_go, multi_episode_entropies = [], [], []
        
        for episode in range(num_episodes_per_epoch):
            obs, info = env.reset(seed=seed + epoch * num_episodes_per_epoch + episode)
            done = False
            log_probs, rewards, entropies = [], [], []
        
            while not done:
                logits = policy(torch.tensor(obs, device=device).float())
                dist = Categorical(logits=logits)
                action = dist.sample()
                entropy = dist.entropy()
                log_prob = dist.log_prob(action)
                # action_distribution = F.softmax(logits, dim=1)
                # entropy = -torch.sum(action_distribution * torch.log(action_distribution + 1e-8))
                # action = torch.multinomial(action_distribution, 1) # Sample an action from action distribution
                # log_prob = torch.log(action_distribution[action]) # Calculate log probability of the action
                next_obs, reward, terminated, truncated, info = env.step(action.item())
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                obs = next_obs
                done = terminated or truncated

            # Compute rewards to go
            rewards_to_go = torch.tensor(compute_rewards_to_go(rewards, gamma=0.99), dtype=torch.float32)
            multi_episode_entropies.extend(entropies)
            multi_episode_rewards_to_go.append(rewards_to_go)
            multi_episode_log_probs.extend(log_probs)
            
        # Subtract baseline 
        multi_episode_rewards_to_go = torch.cat(multi_episode_rewards_to_go)
        advantages = (multi_episode_rewards_to_go - multi_episode_rewards_to_go.mean())/(multi_episode_rewards_to_go.std() + 1e-8)
        
        # Compute loss
        logp = torch.stack(multi_episode_log_probs, device=device)
        ent = torch.stack(multi_episode_entropies, device=device)
        loss = -(logp * advantages).mean() - 0.02 * ent.mean()
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

        # Test policy after 100 episodes
        if (epoch+1) % 100 == 0:
            test_rewards = []   
            policy.eval() 
            with torch.no_grad():      
                for i in range(5):
                    obs, info = env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        logits = policy(torch.tensor(obs).float())
                        action = torch.argmax(logits).item()
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                    test_rewards.append(episode_reward)
            print(f"Epoch {epoch + 1} - Average reward: {np.mean(test_rewards)}")

if __name__ == "__main__":
    train()