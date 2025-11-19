import torch
from torch.nn import Module, Linear, ReLU, Sequential, Dropout
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import copy
import random
from collections import deque

class CriticNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 256)
        self.fc2 = Linear(256, 256)
        self.fc3 = Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 256)
        self.fc2 = Linear(256, 256)   
        self.fc3 = Linear(256, 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # keeps the actions between -1 and 1
        return x

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train():
    # Configs
    num_episodes = 10000
    noise_mean = 0
    noise_std_init = 0.2
    noise_std_min = 0.05
    noise_decay_steps = 50000 # Decay noise over this many steps
    gamma = 0.99
    batch_size = 256
    num_test_runs = 10
    num_episodes_per_test_run = 100
    tau = 0.005 # Standard DDPG tau is usually smaller, e.g. 0.001 or 0.005
    warmup_steps = 1000 # Steps before training starts

    # Initialize the environment
    env = gym.make("LunarLanderContinuous-v3")

    # Initialize the actor and critic networks
    actor = ActorNetwork()
    critic = CriticNetwork()

    # Initialize optimizers
    actor_optimizer = Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = Adam(critic.parameters(), lr=1e-4)

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
    
    global_step = 0
    best_test_reward = 200

    # Training loop
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        # Calculate noise std for this episode (or step)
        # Simple linear decay based on episodes for simplicity, or could be step based
        noise_std = max(noise_std_min, noise_std_init * (1 - global_step / noise_decay_steps))
        noise = Normal(noise_mean, noise_std)

        while not done:
            global_step += 1
            
            # Select action
            with torch.no_grad():
                if global_step < warmup_steps:
                    # Collect random actions during warmup phase 
                    action = env.action_space.sample()
                    action_tensor = torch.tensor(action, dtype=torch.float32)
                else:
                    action_tensor = torch.clamp(
                        actor(torch.tensor(obs, dtype=torch.float32)) + noise.sample(sample_shape=(2,)),
                        -1, 1)
                    action = action_tensor.numpy()

            # Execute action
            next_obs, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated

            # Store transition
            replay_buffer.add(obs, action, reward, next_obs, done)    
            obs = next_obs

            # Update networks if we have enough data
            if len(replay_buffer) > batch_size and global_step > warmup_steps:
                # Sample a random minibatch
                batch = replay_buffer.sample(batch_size=batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.float32)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(np.array(dones), dtype=torch.float32)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_actions = actor_target(next_states)
                    next_state_action = torch.cat([next_states, next_actions], dim=1)
                    next_state_q = critic_target(next_state_action).squeeze()
                    y = rewards + gamma * (1.0 - dones) * next_state_q
                
                # Update critic
                critic_optimizer.zero_grad()
                state_action = torch.cat([states, actions], dim=1)
                critic_loss = F.mse_loss(critic(state_action).squeeze(), y)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5) 
                critic_optimizer.step()
                
                # Update actor
                actor_optimizer.zero_grad()
                actor_loss = -critic(torch.cat([states, actor(states)], dim=1)).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                actor_optimizer.step()

                # Update target networks (Soft Update)
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

        # Test the trained policy every nth epoch
        if (episode+1) % num_episodes_per_test_run == 0:    
            test_rewards = []
            for i in range(num_test_runs):
                test_obs, _ = env.reset()
                test_done = False
                test_episode_reward = 0.0
                while not test_done:
                    with torch.no_grad():
                        test_action = actor(torch.tensor(test_obs, dtype=torch.float32))
                    test_obs, test_reward, test_truncated, test_terminated, _ = env.step(test_action.numpy())
                    test_done = test_truncated or test_terminated
                    test_episode_reward += test_reward
                test_rewards.append(test_episode_reward)
            print(f"Episode {episode+1}, Step {global_step}, Average test reward: {np.mean(test_rewards):.2f}")
            if np.mean(test_rewards) > best_test_reward:
                best_test_reward = np.mean(test_rewards)
                import os
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save(actor.state_dict(), "checkpoints/actor.pth")
                torch.save(critic.state_dict(), "checkpoints/critic.pth")
        

if __name__ == "__main__":
    train()