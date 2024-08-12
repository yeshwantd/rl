import gym, torch
from policy import Policy
import utils, time

def main():
    env = gym.make('LunarLander-v2', new_step_api=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(state_dim, action_dim)
    # observations, actions, rewards, sum_discounted_rewards = utils.get_samples(env, policy, 10)
    num_trajectories = 100
    num_epochs = 100
    optimizer = torch.optim.Adam(params=policy.parameters())
    for i in range(num_epochs):
        t0 = time.time()
        observations, actions, rewards, sum_discounted_rewards = utils.get_samples(env, policy, num_trajectories)
        t1 = time.time()
        policy, loss = utils.train_step(policy, optimizer, observations, actions, sum_discounted_rewards)
        t2 = time.time()
        print(f"{i}: Loss {loss:0.4f}, sampling time {t1 - t0:0.4f}, training time {t2 - t1:0.4f}")

    new_env = gym.make('LunarLander-v2', new_step_api=True, render_mode="human")
    observation = new_env.reset()
    count = 0
    for i in range(1000):
        observation, reward, terminated, truncated, info = new_env.step(policy(torch.tensor(observation)).argmax().item())
        # print(reward, terminated, truncated)
        if terminated or truncated:
            observation = new_env.reset()
            count += 1
            print(f"{count}: ========== Environment reset ==========")
    

if __name__ == '__main__':
    main()