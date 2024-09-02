import gymnasium as gym, torch, numpy as np, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from policy import Policy
from value import Value
import utils

def train(policy_filename, value_filename, policy, value):
    # Hyperparameters
    discount_rate = 0.99
    num_epochs = 5000
    checkpoint_frequency = 500

    # Create the environment
    # env = gym.make('LunarLander-v2', new_step_api=True)
    env = gym.make("LunarLander-v2")
    
    # Initialize the policy if not passed as a parameter
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if policy is None:
        policy = Policy(state_dim, action_dim)
    if value is None:
        value = Value(state_dim)

    policy_optimizer = torch.optim.Adam(params=policy.parameters())
    value_optimizer = torch.optim.Adam(params=value.parameters())
    
    policy_losses, value_losses = utils.train(policy, value, policy_optimizer, value_optimizer, env, num_epochs, discount_rate, policy_filename, value_filename, checkpoint_frequency)
    return policy_losses, value_losses

def display(policy_filename, episodes):
    # load the policy from saved policy.pt file
    policy = Policy(state_dim=8, action_dim=4)
    policy.load_state_dict(torch.load(f"{policy_filename}.pt"))
    policy.eval()
    env = gym.make('LunarLander-v2', render_mode="human")
    observation, _ = env.reset()
    episode_reward = 0
    while episodes > 0:
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).view(1,-1)
            action = policy.action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        
        if terminated or truncated:
            observation, _ = env.reset()
            episodes -= 1
            print(f"{episodes}: Sum reward {episode_reward}")
            episode_reward = 0

if __name__ == '__main__':
    checkpoint_path = "/Users/ydatta/my_workspace/RL/checkpoints/Actor Critic/"
    policy_filename = checkpoint_path + "policy_ac_1"
    value_filename = checkpoint_path + "value_ac_1"
    
    use_checkpoint = False
    if use_checkpoint:
        policy = Policy(state_dim=8, action_dim=4)
        policy.load_state_dict(torch.load(f"{policy_filename}.pt"))
        value = Value(state_dim=8)
        value.load_state_dict(torch.load(f"{value_filename}.pt"))
    else:
        policy = None
        value = None

    train_run = False
    if train_run:
        policy_losses, value_losses = train(policy_filename, value_filename, policy, value)
        print(policy_losses, value_losses)
    else:
        specific_checkpoint = True
        if specific_checkpoint:
            policy_filename = checkpoint_path + "policy_ac_1_5000"
        display(policy_filename, 3)

    
