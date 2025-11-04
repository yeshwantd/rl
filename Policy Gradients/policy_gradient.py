import torch
import numpy as np
import gymnasium as gym

# internal imports
import utils
from models import BasicPolicy,  BasicPolicyWithLayerNorm
from landers import ShapedLunarLander

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRAIN =====
def run_single_episode(policy, env, seed=None):

    observation, info = env.reset(seed=seed)
    terminated_or_truncated = False
    rewards, log_probs, entropies = [], [], []

    while not terminated_or_truncated:

        # Get the probability distribution over all actions
        action_distribution = policy.get_action_distribution(torch.tensor(observation))

        # Compute the entropy
        entropy = -torch.sum(action_distribution * torch.log(action_distribution))
        
        # Pick one action from the distribution based on their probability values
        action = torch.multinomial(action_distribution, 1).item()
        
        # Execute the action and get feedback from environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Get log probabilities of the action taken
        log_prob = torch.log(action_distribution[action])
        
        # Store rewards, log probabilities and entropies
        rewards.append(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Update the observation
        observation = next_observation
        
        # Check if the episode is terminated or truncated
        terminated_or_truncated = terminated or truncated
    
    return log_probs, rewards, entropies



def train_batch(policy, optimizer, env, gamma, batch_size, seed=None, beta=None):
    """
    Train the policy for one batch

    Inputs:
        gamma: discount factor
        beta: weight for the entropy factor in the loss
    """
    # variables
    all_log_probs = []
    all_rewards_to_go = []

    # Collect data for one episode
    rewards = []
    log_probs = []
    
    policy.train()
    for i in range(batch_size):
        log_probs, rewards, entropies = run_single_episode(policy=policy, env=env, seed=seed+i if seed else None)
        discounted_rewards_to_go = utils.compute_rewards_to_go(rewards, gamma)
        all_log_probs.extend(log_probs)
        all_rewards_to_go.extend(discounted_rewards_to_go)

    # Convert lists to tensors
    log_probs_tensor = torch.stack(all_log_probs)
    rewards_to_go_tensor = torch.tensor(all_rewards_to_go, dtype=torch.float32)
    entropies_tensor = torch.stack(entropies)

    # Reduce variance
    rewards_to_go_tensor = (rewards_to_go_tensor - rewards_to_go_tensor.mean()) / (rewards_to_go_tensor.std() + 1e-9)

    # Compute loss
    if beta is not None:
        loss = (1/batch_size)*(-torch.sum(log_probs_tensor * rewards_to_go_tensor) - beta * entropies_tensor.mean())
    else:
        loss = (1/batch_size)*(-torch.sum(log_probs_tensor * rewards_to_go_tensor))

    # Backpropagate the loss
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0) # Gradient clipping
    optimizer.step()
    
    return loss.item()
                
if  __name__ == "__main__":
    
    demo = True # set to true if you want to visualize the policy
    plot = False # set to true if you want to plot the losses and avg. test rewards 

    # Set the seed for reproducability
    set_seed = True
    seed = 100
    if set_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    gamma = 0.99
    beta_start, beta_end = 1e-3, 1e-8
    # beta_start, beta_end = 0, 0
    max_episode_steps = 500
    num_episodes = 4000
    batch_size = 10
    
    # Create Policy
    policy = BasicPolicy()
    # policy = BasicPolicyWithLayerNorm()
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995) 

    env = gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)
    # env = ShapedLunarLander(gym.make("LunarLander-v3", max_episode_steps=max_episode_steps))
    losses = []
    avg_rewards = []
    n_batches = num_episodes//batch_size
    for i in range(n_batches):
        loss = train_batch(
            policy=policy, 
            optimizer=optimizer, 
            env=env, 
            gamma=gamma, 
            batch_size=batch_size, 
            seed=(seed + i*batch_size) if set_seed else None,
            beta=beta_start + (beta_end - beta_start)*(i/(n_batches-1))
        )
        scheduler.step()
        # print(scheduler.get_last_lr())
        losses.append(loss)
        # print(f"Episode {i+1}: Loss = {loss}")

        # Test policy every 100 episodes
        if (i+1) % (100/batch_size) == 0:
            rewards = utils.test_policy(env, policy)
            avg_reward = np.mean(rewards)
            print(f"Episode {i+1}: Average Reward = {avg_reward}")
            if avg_reward > 200:
                torch.save(policy.state_dict(), f"checkpoints/policy_{i+1}.pt")
                avg_rewards.append([(i+1), avg_reward])

    
    if plot:
        # Plot the losses
        utils.plot(losses,  "Episode", "Loss", "Losses")
        utils.plot(avg_rewards, "Episode", "Average Reward", "Average Rewards")
    
    if demo:
        env = gym.make("LunarLander-v3", render_mode="human")
        # Pick the policy with the highest average reward
        best_episode = max(avg_rewards, key=lambda x: x[1])
        print(f"Best Episode: {best_episode[0]} with Average Reward: {best_episode[1]}")
        policy.load_state_dict(torch.load(f"checkpoints/policy_{best_episode[0]}.pt"))
        utils.visualize_policy(env, policy, 5)
    
