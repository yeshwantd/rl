import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

# internal imports
import utils
from models import BasicPolicy,  BasicPolicyWithLayerNorm
from landers import ShapedLunarLander
import configs

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_envs(num_envs, max_episode_steps, seed):
    def make_env(rank):
        def thunk():
            env = gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)
            env.reset(seed = (seed + rank) if seed else None)
            return env
        return thunk
    envs = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    return envs

# ===== TRAIN =====
def collect_data(policy, num_envs, max_episode_steps, seed=None):
    envs = get_envs(num_envs, max_episode_steps, seed)
    obs, infos = envs.reset()
    episodes_completed = np.zeros(num_envs, dtype=bool)

    all_rewards = [[] for _ in range(num_envs)]
    all_log_probs = [[] for _ in range(num_envs)]
    all_entropies = [[] for _ in range(num_envs)]

    while not np.all(episodes_completed):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # Get action probabilities for all envs in batch
        action_distributions = policy.get_action_distribution(obs_t) # shape (N,A)

        # Compute entropy for all envs in batch
        entropies = -torch.sum(action_distributions * torch.log(action_distributions + 1e-8), dim=1, keepdim=True) # shape (N,1)

        # Sample one action per env
        actions = torch.multinomial(action_distributions, num_samples=1) # shape (N, 1)

        # Get log probabilities of the actions taken
        log_probs = torch.log(action_distributions.gather(dim=1, index=actions)) # shape (N, 1)

        # Execute the actions
        next_obs, rewards, terminated, truncated, infos = envs.step(actions.squeeze(1).cpu().numpy())

        for i in range(num_envs):
            if not episodes_completed[i]: 
                all_rewards[i].append(rewards[i])
                all_log_probs[i].append(log_probs[i])
                all_entropies[i].append(entropies[i])

            if terminated[i] or truncated[i]:
                episodes_completed[i] = True

        obs = next_obs

    return all_log_probs, all_rewards, all_entropies

def train(policy, optimizer, gamma, beta, seed):
    rewards_to_go = []

    policy.train()
    log_probs, rewards, entropies = collect_data(policy, configs.num_envs, configs.max_episode_steps, seed)
    for i in range(configs.num_envs):
        discounted_rewards_to_go = utils.compute_rewards_to_go(rewards[i], gamma)
        rewards_to_go.extend(discounted_rewards_to_go)
        
    # Convert lists to tensors on the correct device
    log_probs_tensor = torch.stack([log_probs[i][j] for i in range(configs.num_envs) for j in range(len(log_probs[i]))])
    rewards_to_go_tensor = torch.tensor(rewards_to_go, dtype=torch.float32, device=device).unsqueeze(1)
    entropies_tensor = torch.stack([entropies[i][j] for i in range(configs.num_envs) for j in range(len(entropies[i]))])    

    # Reduce variance
    rewards_to_go_tensor = (rewards_to_go_tensor - rewards_to_go_tensor.mean()) / (rewards_to_go_tensor.std() + 1e-9)

    # Compute loss
    loss = (1/configs.num_envs) * (-torch.sum(log_probs_tensor * rewards_to_go_tensor)) - beta * entropies_tensor.mean()

    # Backpropagate the loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0) # Gradient clipping
    optimizer.step()
    
    return loss.item()

                
if  __name__ == "__main__":
    
    demo = True # set to true if you want to visualize the policy
    plot = True # set to true if you want to plot the losses and avg. test rewards 
    
    if configs.set_seed:
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.cuda.manual_seed_all(configs.seed)
    
    # Create Policy and move to device
    policy = BasicPolicy().to(device)
    # policy = BasicPolicyWithLayerNorm().to(device)
    
    print(f"Using device: {device}")
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995) 

    env = gym.make("LunarLander-v3", max_episode_steps=configs.max_episode_steps) # for testing
    # env = ShapedLunarLander(gym.make("LunarLander-v3", max_episode_steps=max_episode_steps))
    avg_rewards = []
    max_avg_test_reward = 0

    training_loss, losses = [], []
    for epoch in range(configs.num_epochs):
        loss = train(
            policy=policy,
            optimizer=optimizer,
            gamma=configs.gamma,
            beta=configs.beta_start - ((configs.beta_start - configs.beta_end)*epoch/configs.num_epochs),
            seed=configs.seed + epoch if configs.set_seed else None
        )
        scheduler.step()
        losses.append(loss)
        
        # Test policy every 100 episodes
        if (epoch+1) % 100 == 0:
            rewards = utils.test_policy(env, policy, device=device)
            avg_reward = np.mean(rewards)
            print(f"Epoch {epoch+1}: Training Loss = {np.mean(losses)}, Average Reward = {avg_reward}")
            training_loss.append(np.mean(losses))
            losses = [] # reset losses
            if avg_reward > 200:
                torch.save(policy.state_dict(), f"checkpoints/policy_{epoch+1}.pt")
                avg_rewards.append([(epoch+1), avg_reward])
                if avg_reward > max_avg_test_reward:
                    max_avg_test_reward = avg_reward
                    torch.save(policy.state_dict(), f"checkpoints/best_policy.pt")
    
    if plot:
        # Plot the losses and average rewards
        utils.plot(training_loss,  "Episode", "Loss", "Losses")
        utils.plot(avg_rewards, "Episode", "Average Reward", "Average Rewards")
    
    if demo:
        env = gym.make("LunarLander-v3", render_mode="human")
        # Pick the policy with the highest average reward
        policy.load_state_dict(torch.load(f"checkpoints/best_policy.pt", map_location=device))
        utils.visualize_policy(env, policy, 5, device=device)
    
