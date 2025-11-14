import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from models import BasicPolicy, BasicValue
from utils import test_policy, visualize_policy
import config

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
    Train a critic model using the given episodes and optimizer with batched updates.

    Args:
        model (nn.Module): The critic model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        episodes (list): A list of episodes, where each episode is a list of (observation, action, reward, next_observation) tuples.
        gamma (float): The discount factor.
        device: The device to run on.

    Returns:
        float: The average loss over all episodes.
    """
    # Accumulate loss across all episodes, then do single update
    total_loss = 0.0
    num_transitions = 0

    for episode in episodes:
        obs = torch.from_numpy(np.array([t[0] for t in episode], dtype=np.float32)).to(device)
        rewards = torch.from_numpy(np.array([t[2] for t in episode], dtype=np.float32)).to(device)
        next_obs = torch.from_numpy(np.array([t[3] for t in episode], dtype=np.float32)).to(device)

        # TD target: r + gamma * V(s_{t+1})
        targets = rewards + gamma * model(next_obs).detach().squeeze(-1)
        predictions = model(obs).squeeze(-1)
        
        # Accumulate loss (sum, not mean, so we can average across all transitions later)
        total_loss += F.huber_loss(predictions, targets, reduction='sum')
        num_transitions += len(episode)

    # Single gradient update for all episodes
    avg_loss = total_loss / num_transitions
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return avg_loss.item()

# Train the actor model via policy gradient
def train_actor(actor, critic, optimizer, episodes, gamma, device, entropy_coef=0.01, grad_clip=1.0):
    """
    Train the actor model using the given episodes and optimizer with batched updates and variance reduction.

    Args:
        actor (nn.Module): The actor model to train.
        critic (nn.Module): The critic model for advantage estimation.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        episodes (list): A list of episodes, where each episode is a list of (observation, action, reward, next_observation) tuples.
        gamma (float): The discount factor.
        device: The device to run on.
        entropy_coef (float): Coefficient for entropy regularization to encourage exploration.
        grad_clip (float): Maximum gradient norm for gradient clipping.

    Returns:
        float: The total loss value.
    """
    # Collect all advantages for normalization
    all_advantages = []
    episode_data = []
    
    for episode in episodes:
        obs = torch.from_numpy(np.array([t[0] for t in episode], dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array([t[1] for t in episode], dtype=np.int64)).to(device)
        rewards = torch.from_numpy(np.array([t[2] for t in episode], dtype=np.float32)).to(device)
        next_obs = torch.from_numpy(np.array([t[3] for t in episode], dtype=np.float32)).to(device)

        with torch.no_grad():
            advantages = rewards + gamma * critic(next_obs).squeeze(-1) - critic(obs).squeeze(-1)
        
        all_advantages.append(advantages)
        episode_data.append((obs, actions))
    
    # Normalize advantages across all episodes for variance reduction
    all_advantages_cat = torch.cat(all_advantages)
    adv_mean = all_advantages_cat.mean()
    adv_std = all_advantages_cat.std() + 1e-8
    
    # Accumulate loss across all episodes, then do single update
    total_policy_loss = 0.0
    total_entropy = 0.0
    num_transitions = 0
    
    for i, (obs, actions) in enumerate(episode_data):
        action_probs = actor.get_action_distribution(obs)
        chosen_action_probs = action_probs.gather(1, actions.view(-1, 1)).squeeze(1).clamp(min=1e-8)
        
        # Normalize advantages
        normalized_advantages = (all_advantages[i] - adv_mean) / adv_std
        
        # Policy gradient loss (sum across transitions)
        total_policy_loss += -torch.sum(torch.log(chosen_action_probs) * normalized_advantages)
        
        # Entropy bonus for exploration (sum across transitions)
        total_entropy += -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        
        num_transitions += len(obs)
    
    # Average loss across all transitions
    avg_policy_loss = total_policy_loss / num_transitions
    avg_entropy = total_entropy / num_transitions
    
    # Total loss = policy loss - entropy bonus
    total_loss = avg_policy_loss - entropy_coef * avg_entropy
    
    # Single gradient update for all episodes
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    optimizer.step()
    
    return total_loss.item()


# Train the actor and critic models
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # vectorized env for training data
    # venv = SyncVectorEnv([make_env_fn("LunarLander-v3", config.max_episode_steps, config.seed + i) for i in range(config.num_envs)])   
    venv = AsyncVectorEnv([make_env_fn("LunarLander-v3", config.max_episode_steps, config.seed + i) for i in range(config.num_envs)])
    # single env for eval/visualization
    eval_env = gym.make("LunarLander-v3", max_episode_steps=config.max_episode_steps)

    actor = BasicPolicy().to(device)
    critic = BasicValue().to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)

    critic_losses = []
    # warm start critic using parallel rollouts with diverse trajectories
    for i in range(config.num_critic_warm_start_epochs):
        # Use different seed each warm start epoch for diversity
        warmup_seed = config.seed + i if config.seed is not None else None
        # Use eval_mode=True since we're not training the actor (saves memory, no gradients needed)
        episodes = generate_episodic_data(venv, actor, config.num_episodes, seed=warmup_seed,
                                          eval_mode=True, device=device)
        critic_loss = train_critic(critic, critic_optimizer, episodes, config.gamma, device)
        critic_losses.append(critic_loss)
        print(f"Warm start epoch: {i}, Critic Loss: {critic_loss}")

    if config.SHOW_PLOTS:
        plt.plot(critic_losses, label="Critic Loss")
        plt.show()

    actor_losses, critic_losses = [], []
    best_rewards = -np.inf
    for epoch in range(config.num_epochs):
        # Use different seed each epoch for diverse experiences
        epoch_seed = config.seed + epoch if config.seed is not None else None
        episodes = generate_episodic_data(venv, actor, config.num_episodes, seed=epoch_seed,
                                          eval_mode=False, device=device)
        
        # Decay entropy coefficient linearly from start to end value
        if epoch < config.entropy_coef_decay_epochs:
            entropy_coef = config.entropy_coef_start - (config.entropy_coef_start - config.entropy_coef_end) * (epoch / config.entropy_coef_decay_epochs)
        else:
            entropy_coef = config.entropy_coef_end
        
        # Update critic once with batched update across all episodes
        critic_loss = train_critic(critic, critic_optimizer, episodes, config.gamma, device)
        
        # Update actor once with batched update across all episodes
        actor_loss = train_actor(actor, critic, actor_optimizer, episodes, config.gamma, device, 
                                entropy_coef, config.grad_clip)
        
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        print(f"Epoch: {epoch}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}, Entropy Coef: {entropy_coef:.4f}")

        if (epoch + 1) % 10 == 0:
            avg_rewards = np.mean(test_policy(eval_env, actor, device=device))
            print(f"Average rewards for policy after {epoch+1} epochs: {avg_rewards}")
            if avg_rewards > best_rewards:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(actor.state_dict(), "checkpoints/best_policy.pt")
                best_rewards = avg_rewards

    if config.SHOW_PLOTS:
        plt.plot(actor_losses, label="Actor Loss")
        plt.plot(critic_losses, label="Critic Loss")
        plt.legend(); plt.show()

    # visualize best policy in a human-rendered single env
    actor.load_state_dict(torch.load("checkpoints/best_policy.pt"))
    human_env = gym.make("LunarLander-v3", render_mode="human")
    visualize_policy(human_env, actor, device=device)

if __name__ == "__main__":
    main()