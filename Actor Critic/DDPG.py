import torch
from torch.nn import Module, Linear, ReLU, Sequential, Dropout
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import copy
import random

class CriticNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 64)
        self.fc2 = Linear(64, 64)   
        self.fc3 = Linear(64, 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # keeps the actions between -1 and 1
        return x

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size if batch_size < len(self.buffer) else len(self.buffer))

def train():
    # Configs
    num_episodes = 10000
    noise_mean = 0
    noise_std = 0.2
    gamma = 0.99
    batch_size = 128
    num_test_runs = 10
    num_episodes_per_test_run = 1000
    tau = 0.001

    # Initialize the environment
    env = gym.make("LunarLanderContinuous-v3")

    # Initialize the actor and critic networks
    actor = ActorNetwork()
    critic = CriticNetwork()

    # Initialize target networks with the same weights as the original networks
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    
    # Disable gradients for target networks
    for param in actor_target.parameters():
        param.requires_grad = False
    for param in critic_target.parameters():
        param.requires_grad = False

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size=100000)

    # Initialize optimizers
    actor_optimizer = Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = Adam(critic.parameters(), lr=0.001)

    # Training loop
    for episode in range(num_episodes):
        # Initilaize random process N for action exploration
        noise = Normal(noise_mean, noise_std)

        # Reset the environment and get the initial state
        obs, info = env.reset()
        done = False

        # Collect all steps from an episode into the buffer
        while not done:
            # Select action according to current policy and exploration noise
            with torch.no_grad():
                action = actor(torch.tensor(obs, dtype=torch.float32)) + noise.sample(sample_shape=(2,))        
            # Execute action and observe reward and next state
            next_obs, reward, truncated, terminated, info = env.step(action.numpy())
            # Store transition in replay buffer
            done = truncated or terminated
            replay_buffer.add(obs, action, reward, next_obs, done)    
            obs = next_obs

        for _ in range(64):
            # Sample a random minibatch of transitions from replay buffer
            batch = replay_buffer.sample(batch_size=batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(np.array(dones), dtype=torch.float32)
            
            # Compute target Q-values
            next_actions = actor_target(next_states)
            next_state_action = torch.cat([next_states, next_actions], dim=1)
            y = rewards + gamma * (1.0 - dones) * critic_target(next_state_action).squeeze()
            
            # Update critic
            critic_optimizer.zero_grad(set_to_none=True)
            state_action = torch.cat([states, actions], dim=1)
            critic_loss = F.mse_loss(critic(state_action).squeeze(), y)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5)
            critic_loss.backward()
            critic_optimizer.step()
            
            # Update actor
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss = -critic(torch.cat([states, actor(states)], dim=1)).mean()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5)
            actor_loss.backward()
            actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

        # Test the trained policy every nth epoch
        if (episode+1) % num_episodes_per_test_run == 0:    
            test_rewards = []
            for i in range(num_test_runs):
                obs, info = env.reset()
                done = False
                episode_reward = 0.0
                while not done:
                    with torch.no_grad():
                        action = actor(torch.tensor(obs, dtype=torch.float32))
                    obs, reward, truncated, terminated, info = env.step(action.numpy())
                    done = truncated or terminated
                    episode_reward += reward
                test_rewards.append(episode_reward)
            print(f"Average test reward: {np.mean(test_rewards)}")
        
if __name__ == "__main__":
    train()


        

