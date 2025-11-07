import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from models import BasicPolicy, BasicValue
from utils import test_policy, visualize_policy

# Factory function to return a function that creates a single environment to be used by AsyncVectorEnv
def make_env_fn(env_id="LunarLander-v3", max_episode_steps=500, seed_offset=0):
    def thunk():
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        # seed on first reset; gymnasium handles per-env RNG afterwards
        env.reset(seed=seed_offset)
        return env
    return thunk

# Generate episodic data for a given policy
def generate_episodic_data(venv, policy, num_episodes, seed=None, eval_mode=False, device='cpu'):
    """
    Collect episodic rollout data using a vectorized environment.

    This function runs multiple environments in parallel (via an AsyncVectorEnv),
    samples actions from the given policy, and records full episodes until the 
    specified number of completed episodes is reached.

    Args:
        venv (gym.vector.AsyncVectorEnv): 
            The vectorized environment containing N copies of the base environment
            running in parallel processes.
        policy (nn.Module): 
            The actor network that outputs a probability distribution over actions
            for a given state.
        num_episodes (int): 
            The total number of complete episodes to collect across all environments.
        seed (int, optional): 
            Random seed for deterministic rollouts. Defaults to None.
        eval_mode (bool, optional): 
            If True, sets the policy to evaluation mode and disables gradient tracking. 
            If False, enables training mode with gradient computation. Defaults to False.
        device (str, optional): 
            The computation device to use ("cuda", "mps", or "cpu"). Defaults to "cpu".

    Returns:
        list[list[tuple]]: 
            A list of completed episodes, where each episode is a list of tuples 
            in the format:
            (observation, action, reward, next_observation, action_probability).
    """

    episodes = []
    if eval_mode:
        policy.eval()
        for param in policy.parameters():
            param.requires_grad = False
    else:
        policy.train()
        for p in policy.parameters():
            p.requires_grad = True

    # reset all envs
    if seed is not None:
        obs, infos = venv.reset(seed=seed)
    else:
        obs, infos = venv.reset()

    num_envs = obs.shape[0]
    buffers = [[] for _ in range(num_envs)]  # per-env step buffers

    def _get_action_probs(obs_np):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.set_grad_enabled(not eval_mode):
            dist = policy.get_action_distribution(obs_t)
            if dist.dim() == 1:  # (A,) → not batched
                probs = []
                for i in range(obs_t.shape[0]):
                    di = policy.get_action_distribution(obs_t[i])
                    probs.append(di)
                probs = torch.stack(probs, dim=0)
            else:
                probs = dist  # (N, A)
        return probs
    
    while len(episodes) < num_episodes:
        probs = _get_action_probs(obs)  # (N, A)
        # sample one action per env
        actions_t = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (N,)
        chosen_probs_t = probs.gather(1, actions_t.view(-1, 1)).squeeze(1)

        actions = actions_t.detach().cpu().numpy()
        chosen_probs = chosen_probs_t.detach().cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = venv.step(actions)

        # record transitions per env
        for i in range(num_envs):
            buffers[i].append((
                obs[i].copy(),
                int(actions[i]),
                float(rewards[i]),
                next_obs[i].copy(),
                float(chosen_probs[i]),
            ))
            if terminated[i] or truncated[i]:
                episodes.append(buffers[i])
                buffers[i] = []
                if len(episodes) >= num_episodes:
                    # we have enough episodes; stop early without resetting
                    break
        obs = next_obs

    return episodes[:num_episodes]

# Train a critic model
def train_critic(model, optimizer, episodes, gamma, device):
    """
    Train a critic model using the given episodes and optimizer.

    Args:
        model (nn.Module): The critic model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        episodes (list): A list of episodes, where each episode is a list of (observation, action, reward, next_observation) tuples.
        gamma (float): The discount factor.

    Returns:
        float: The average loss over all episodes.
    """
    losses = []
    for episode in episodes:
        loss = 0
        for i, (observation, action, reward, next_observation, action_prob) in enumerate(episode):
            observation = torch.tensor(observation, dtype=torch.float32).to(device)
            next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
            # y_t = r(s_t,a_t) + γ * V(s_{t+1})
            target = reward + gamma * model(next_observation).detach()
            prediction = model(observation)
            loss += F.mse_loss(target, prediction)
        
        # Gradient descent on accumulated losses
        optimizer.zero_grad()
        loss = loss / len(episode)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# Train the actor model via policy gradient
def train_actor(actor, critic, optimizer, episodes, gamma, device):
    """
    Train the actor model using the given episodes and optimizer.

    Args:
        model (nn.Module): The actor model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        episodes (list): A list of episodes, where each episode is a list of (observation, action, reward, next_observation) tuples.
        gamma (float): The discount factor.

    Returns:
        float: The average loss over all episodes.
    """
    losses = []
    for episode in episodes:
        loss = 0
        for i, (observation, action, reward, next_observation, _) in enumerate(episode):
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
            next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32).to(device)
            
            # Recompute action probability with current policy parameters (fixes computational graph issue)
            action_distribution = actor.get_action_distribution(obs_tensor)
            action_prob = action_distribution[action]
            
            advantage = reward + gamma * critic(next_obs_tensor) - critic(obs_tensor)
            loss += -torch.log(action_prob) * advantage
        
        # Gradient descent on accumulated losses over episode
        optimizer.zero_grad()
        loss = loss / len(episode)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


# Train the actor and critic models
def main():
    SHOW_PLOTS = False

    num_epochs = 4000
    num_critic_warm_start_epochs = 100
    num_episodes = 10
    gamma = 0.99
    seed = 42
    max_episode_steps = 500
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    num_envs = 8  # parallel envs for rollouts
    # vectorized env for training data
    venv = AsyncVectorEnv([make_env_fn("LunarLander-v3", max_episode_steps, seed + i)
                           for i in range(num_envs)])
    # single env for eval/visualization
    eval_env = gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)

    actor = BasicPolicy().to(device)
    critic = BasicValue().to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-1)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    critic_losses = []
    # warm start critic using parallel rollouts (policy in eval mode)
    for i in range(num_critic_warm_start_epochs):
        episodes = generate_episodic_data(venv, actor, num_episodes, seed=seed,
                                          eval_mode=True, device=device)
        critic_loss = train_critic(critic, critic_optimizer, episodes, gamma, device)
        critic_losses.append(critic_loss)
        print(f"Warm start epoch: {i}, Critic Loss: {critic_loss}")

    if SHOW_PLOTS:
        plt.plot(critic_losses, label="Critic Loss")
        plt.show()

    actor_losses, critic_losses = [], []
    best_rewards = -np.inf
    for epoch in range(num_epochs):
        episodes = generate_episodic_data(venv, actor, num_episodes, seed=seed,
                                          eval_mode=False, device=device)
        critic_loss = train_critic(critic, critic_optimizer, episodes, gamma, device)
        actor_loss = train_actor(actor, critic, actor_optimizer, episodes, gamma, device)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        print(f"Epoch: {epoch}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

        if (epoch + 1) % 10 == 0:
            avg_rewards = np.mean(test_policy(eval_env, actor))
            print(f"Average rewards for policy after {epoch+1} epochs: {avg_rewards}")
            if avg_rewards > best_rewards:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(actor.state_dict(), "checkpoints/best_policy.pt")
                best_rewards = avg_rewards

    if SHOW_PLOTS:
        plt.plot(actor_losses, label="Actor Loss")
        plt.plot(critic_losses, label="Critic Loss")
        plt.legend(); plt.show()

    # visualize best policy in a human-rendered single env
    actor.load_state_dict(torch.load("checkpoints/best_policy.pt"))
    human_env = gym.make("LunarLander-v3", render_mode="human")
    visualize_policy(human_env, actor)

if __name__ == "__main__":
    main()