import gymnasium as gym

class ShapedLunarLander(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs = [x, y, x_dot, y_dot, angle, angular_vel, leg_contact_left, leg_contact_right]
        x, y, x_dot, y_dot, angle, ang_vel, left_leg, right_leg = obs

        shaped_reward = reward
        # shaped_reward = reward \
        #     - 0.1 * abs(x)      \
        #     - 0.1 * abs(y_dot)  \
        #     - 0.1 * abs(angle) 

        # # Bonus reward for landing close to center
        # if abs(x) < 0.1:
        #     shaped_reward += 0.1   # center bonus each step
        if terminated and (abs(x) < 0.05):
            shaped_reward += 10.0  # center bonus for landing
        if terminated and (abs(y) < 0.05):
            shaped_reward += 10.0   # center bonus for landing 
        if terminated and (abs(angle) < 0.01):
            shaped_reward += 10.0   # center bonus for landing
        if terminated and (abs(x_dot) < 0.01):
            shaped_reward += 10.0   # center bonus for landing
        if terminated and (abs(y_dot) < 0.01):
            shaped_reward += 10.0   # center bonus for landing
        if terminated and (abs(ang_vel) < 0.01):
            shaped_reward += 10.0   # center bonus for landing
        if  terminated and (left_leg == 1) and (right_leg == 1):
            shaped_reward += 10.0   # center bonus for landing

        return obs, shaped_reward, terminated, truncated, info