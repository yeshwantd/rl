import gym, torch, numpy as np, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from policy import Policy
import utils, time

def main(policy_filename):
    env = gym.make('LunarLander-v2', new_step_api=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    envs = [gym.make('LunarLander-v2', new_step_api=True) for i in range(10)]
    policy = Policy(state_dim, action_dim)
    num_trajectories = 1024
    time_steps_per_trajectory = 1024
    num_epochs = 200
    optimizer = torch.optim.Adam(params=policy.parameters())
    losses = []
    for i in range(num_epochs):
        t0 = time.time()
        observations, actions, rewards, sum_discounted_rewards = utils.get_samples_parallel(envs, policy, num_samples=num_trajectories, time_steps=time_steps_per_trajectory)
        t1 = time.time()
        policy, loss = utils.train_step(policy, optimizer, observations, actions, sum_discounted_rewards)
        t2 = time.time()
        print(f"{i}: Loss {loss:0.4f}, sampling time {t1 - t0:0.4f}, training time {t2 - t1:0.4f}")
        losses.append(loss)
        # save checkpoint after every 10 epochs
        if i % 20 == 0 and i != 0:
            torch.save(policy.state_dict(), f"{policy_filename}_{i}.pt")

    torch.save(policy.state_dict(), f"{policy_filename}.pt")
    return losses
    
def run(policy_filename, steps):
    # load the policy from saved policy.pt file
    policy = Policy(state_dim=8, action_dim=4)
    policy.load_state_dict(torch.load(f"{policy_filename}.pt"))
    policy.eval()
    env = gym.make('LunarLander-v2', new_step_api=True, render_mode="human")
    observation = env.reset()
    count = 0
    for i in range(steps):
        with torch.no_grad():
            observation, reward, terminated, truncated, info = env.step(policy(torch.tensor(observation)).argmax().item())
        
        if terminated or truncated:
            observation = env.reset()
            count += 1
            print(f"{count}: ========== Environment reset ==========")

def get_samples():
    envs = [gym.make('LunarLander-v2', new_step_api=True) for i in range(10)] 
    policy = Policy(state_dim=8, action_dim=4)
    # policy.load_state_dict(torch.load("policy.pt"))
    # policy.eval()
    return utils.get_samples_parallel(envs, policy, num_samples=10, time_steps=12)

def get_trajectory():
    env = gym.make('LunarLander-v2', new_step_api=True)
    policy = Policy(state_dim=8, action_dim=4)
    # policy.load_state_dict(torch.load("policy.pt"))
    # policy.eval()
    return utils.get_trajectory(env, policy, time_steps=12, discount=0.9)

if __name__ == '__main__':
    policy_filename = "policy_5_100"
    # losses = main(policy_filename)
    # print(losses)
    run(policy_filename, 4000)
    # o, a, r, dr = get_trajectory()
    # print(a, r, dr)
    
