# commenting code based on docs

import gymnasium as gym

# environment is cerated using make() render_mode specifies how the env should be visualized
env = gym.make("LunarLander-v3", render_mode="human")

# this gets a first observation of the environment
observation, info = env.reset()

# 
episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    # env.step executes the selected action to update the environment can be imagined as the robot pressing the controller 
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()