import gymnasium as gym
# remember this one is for progress bars
import tqdm as tqdm
import agent as cartagent
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.1
n_episodes = 30_000
start_epsilon = 1.0
# epsilon_decay = start_epsilon - 0.01
# epsilon_decay = start_epsilon / (n_episodes / 2)
epsilon_decay = start_epsilon * 0.95
# epsilon_decay = (start_epsilon - .0001) / 2
# epsilon_decay = (start_epsilon - .001) / (n_episodes * 0.5)



final_epsilon = 0.001

env = gym.make("CartPole-v1", render_mode=None)
# only un comment this if we want to watch
# env.render() 

# this line right here is what caused the ram spike in the black jack game I think. Set the buffer lower so it doesn't store everything
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = cartagent.CartpoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

# now we are gonna implement the training loop remember we wrap it in tqdm so we get the progress bar

for episode in tqdm.tqdm(range(n_episodes)):
    
    # remeber obs is the game state and info is metadata about the game like seed and timelimit n stuff like that so when we set these values to env.reset it gives the net a new game to play
    obs, info = env.reset()

    # we use this boolean so we know when the game is over
    done = False

    while not done:
        # get an action based on the current game state
        action = agent.get_action(obs)

        # take the action and receive the results. the line below I will likely comment out because this is from black jack and im guessing the cartpole env probably outputs different values 
        # turns out its the same true means the task ends due to completion or failure, truncated ends due to time limit in cart it is 500 steps
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = agent.discretize(next_obs)

        pole_angle = obs[2]

        # Reward shaping: bonus for being near vertical
        angle_bonus = (1.0 - abs(pole_angle) / 0.418) ** 5 # Normalize: 0.418 is max angle before done

        # Optional: clip between 0 and 1 so we don’t go negative
        angle_bonus = max(angle_bonus, 0)

        if angle_bonus == 0:
            reward -= 5

        # Combine it with the base reward
        reward += angle_bonus

        if truncated:
            reward += 5

        if terminated:
            reward -= 10

        # updates the agent based on the result of the action taken and the q-values and reward n stuff
        agent.update(obs, action, reward, terminated, next_state)

        # update the current state observation
        obs = next_obs

        done = terminated or truncated
    
    agent.decay_epsilon()

def get_moving_avgs(arr, window, mode="valid"):
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=mode) / window

rolling_length = 100  # you can use 500 later, but 100 gives quicker feedback early on
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

# 1. Episode rewards
axs[0].set_title("Episode Rewards")
axs[0].plot(get_moving_avgs(env.return_queue, rolling_length, "valid"))
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Total Reward")

# 2. Episode lengths (usually same as reward in CartPole since reward=1 per timestep)
axs[1].set_title("Episode Lengths")
axs[1].plot(get_moving_avgs(env.length_queue, rolling_length, "valid"))
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Steps Survived")

# 3. Training error over time (assuming you're appending to agent.training_error)
axs[2].set_title("Training Error")
axs[2].plot(get_moving_avgs(agent.training_error, rolling_length, "same"))
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Q-Update Error")

plt.tight_layout()
plt.show()

    
