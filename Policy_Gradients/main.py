import gymnasium as gym, torch
from policy import Policy
import utils

def train(checkpoint_filepath, policy):
    # Hyperparameters
    discount = 0.99
    num_epochs = 5000
    checkpoint_frequency = 500
    
    # Create the environment
    env = gym.make('LunarLander-v2')
    
    # Initialize the policy if not passed as a parameter
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if policy is None:
        policy = Policy(state_dim, action_dim)

    policy_optimizer = torch.optim.Adam(params=policy.parameters())

    policy_losses = utils.train(env, policy, policy_optimizer, num_epochs, discount, checkpoint_filepath, checkpoint_frequency, verbose=True)

    return policy_losses

def display(policy_filename, num_episodes):
    policy = Policy(state_dim=8, action_dim=4)
    policy.load_state_dict(torch.load(policy_filename))
    env = gym.make('LunarLander-v2', render_mode='human')
    episode_rewards = utils.run_episode(env, policy, num_episodes, verbose=True)
    return episode_rewards

if __name__ == '__main__':
    policy_path = "/Users/ydatta/my_workspace/RL/checkpoints/Policy Gradient/"
    policy_name = "policy_gradient"
    training_mode = False
    if training_mode:
        policy = None
        _ = train(policy_path + policy_name, policy)
    else:
        policy_name = "policy_gradient_5000.pt"
        _ = display(policy_path + policy_name, 3)