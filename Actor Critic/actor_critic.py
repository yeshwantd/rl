import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

# Create buffer
class Buffer(Dataset):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def add(self, observation, action, reward, next_observation, action_prob):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((observation, action, reward, next_observation, action_prob))
        else:
            self.buffer.pop(0)
            self.buffer.append((state, action, reward, next_observation, action_prob))

# Collect data from a policy and store in buffer
def collect_data(env, policy, buffer):
    observation = env.reset()
    done = False
    while not done:
        action_distribution = policy.get_action_distribution(observation)
        action, action_prob = torch.argmax(action_distribution), torch.max(action_distribution)
        next_observation, reward, terminated, truncated, info = env.step(action)
        buffer.add(observation, action, reward, next_observation, action_prob)
        observation = next_observation
        done = terminated or truncated

# Train the critic model
def train_critic(critic, critic_optimizer, buffer, gamma):
    critic_optimizer.zero_grad()
    loss = 0
    for observation, action, reward, next_observation, action_prob in buffer:
        loss += (reward + gamma * critic.get_value(next_observation) - critic.get_value(observation)) ** 2
    loss.backward()
    critic_optimizer.step()