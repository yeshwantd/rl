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
import os
import sys

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
    num_episodes = 1000
    noise_mean = 0
    noise_std_init = 0.2
    noise_std_min = 0.05
    noise_decay_steps = 50000 # Decay noise over this many steps
    gamma = 0.99
    batch_size = 256
    num_test_runs = 10
    num_episodes_per_test_run = 100
    tau = 0.001 
    warmup_steps = 1000 # Steps before training starts
    train_seed = 42
    test_seed = 55
    max_episode_steps = 500
    algorithm = "--algorithm" in sys.argv
    td3 = True if algorithm == "td3" else False

    # Reproducibility
    if train_seed is not None:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        random.seed(train_seed)

    # Initialize the environment
    env = gym.make("LunarLanderContinuous-v3", max_episode_steps=max_episode_steps)

    # Initialize the actor and critic networks
    actor = ActorNetwork()
    critic = CriticNetwork()
    if td3:
        critic2 = CriticNetwork()

    # Initialize optimizers
    actor_optimizer = Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = Adam(critic.parameters(), lr=1e-4)
    if td3:
        critic2_optimizer = Adam(critic2.parameters(), lr=1e-4)

    # Initialize target networks with the same weights as the original networks
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    if td3:
        critic2_target = copy.deepcopy(critic2)
    
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
        obs, info = env.reset(seed=train_seed + episode if train_seed is not None else None)
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
                    if td3:
                        next_actions = actor_target(next_states) + noise.sample(sample_shape=(batch_size, 2))
                        next_actions = torch.clamp(next_actions, -1, 1)
                    else:
                        next_actions = actor_target(next_states)
                    next_state_action = torch.cat([next_states, next_actions], dim=1)
                    if td3:
                        y = rewards + gamma * (1.0 - dones) * torch.min(critic_target(next_state_action).squeeze(), critic2_target(next_state_action).squeeze())
                    else:
                        y = rewards + gamma * (1.0 - dones) * critic_target(next_state_action).squeeze()
                
                # Update critic
                critic_optimizer.zero_grad()
                state_action = torch.cat([states, actions], dim=1)
                critic_loss = F.mse_loss(critic(state_action).squeeze(), y)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5) 
                critic_optimizer.step()
                if td3:
                    critic2_optimizer.zero_grad()
                    critic2_loss = F.mse_loss(critic2(state_action).squeeze(), y)
                    critic2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=0.5) 
                    critic2_optimizer.step()
                
                # Update actor. If TD3, update it half as often as critic
                if td3:
                    if global_step % 2 == 0:
                        actor_optimizer.zero_grad()
                        actor_loss = -critic(torch.cat([states, actor(states)], dim=1)).mean()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                        actor_optimizer.step()
                else:
                    actor_optimizer.zero_grad()
                    actor_loss = -critic(torch.cat([states, actor(states)], dim=1)).mean()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    actor_optimizer.step()

                # Update target networks using polyak averaging
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

        # Test the trained policy every nth epoch
        if (episode+1) % num_episodes_per_test_run == 0:    
            test_rewards = []
            for i in range(num_test_runs):
                test_obs, _ = env.reset(seed=test_seed + i if test_seed is not None else None)
                test_done = False
                test_episode_reward = 0.0
                while not test_done:
                    with torch.no_grad():
                        test_action = actor(torch.tensor(test_obs, dtype=torch.float32))
                    test_obs, test_reward, test_truncated, test_terminated, _ = env.step(test_action.numpy())
                    test_done = test_truncated or test_terminated
                    test_episode_reward += test_reward
                test_rewards.append(test_episode_reward)
            print(f"Episode {episode+1}, Step {global_step}, {"TD3" if td3 else "DDPG"}, Average test reward: {np.mean(test_rewards):.2f}")
            if np.mean(test_rewards) > best_test_reward:
                best_test_reward = np.mean(test_rewards)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                actor_checkpoint_path = os.path.join(script_dir, "checkpoints", "actor_" + "td3" if td3 else "ddpg" + ".pth")
                critic_checkpoint_path = os.path.join(script_dir, "checkpoints", "critic_" + "td3" if td3 else "ddpg" + ".pth")
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save(actor.state_dict(), actor_checkpoint_path)
                torch.save(critic.state_dict(), critic_checkpoint_path)
        
# Demo the policy
def demo(policy, num_times):
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    for i in range(num_times):    
        obs, info = env.reset()
        done = False
        tot_reward = 0
        policy.eval()
        with torch.no_grad():
            while not done:
                action = policy(torch.tensor(obs, dtype=torch.float32)).numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                tot_reward += reward
                done = terminated or truncated
            print(f"Total reward for episode {i+1}: {tot_reward}")


if __name__ == "__main__":
    train_flag = False
    render_flag = True
    if train_flag:
        train()
    elif render_flag:
        actor = ActorNetwork()
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, "checkpoints", "actor.pth")
        actor.load_state_dict(torch.load(checkpoint_path))
        demo(actor, 5)