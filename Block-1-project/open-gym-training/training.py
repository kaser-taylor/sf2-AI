import gymnasium as gym
# this one adds progress bars
import tqdm as tqdm 
import bagent as bagent
import matplotlib.pyplot as plt
import numpy as np



learning_rate = 0.0001
n_episodes = 1_000_000
start_epsilon = 1.0
# so in the flappy bird qlearning video he decayed by multiplying by .995 gpt says there are pros and cons to doing it linearly like they have here you know exactly when epsilon will reach 0 and gives you a hard switch from exploration to exploitation and its more predictable but doesn't allow exploration towards the end
epsilon_decay = start_epsilon / (n_episodes / 2)
# unless you do this
final_epsilon = 0.1

# sab defines the rules of the game true = sutton and barto book rules false = open ai rules
env = gym.make("Blackjack-v1", sab=False)
# tracks stats about the episodes so you can plot and get artsy with the hyperparams
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = bagent.BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon, 
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon

)

# This training loop is provided by gpt and I will get a line by line for it from there too

for episode in tqdm.tqdm(range(n_episodes)):
    # so at the beginning of every episode we reinitialize the environment basically giving the agent a new game to play
    obs, info = env.reset()
    # when done = true the while loop stops then we decay. an example is a win a loss or something like that
    done = False

    while not done: 
        # agent chooses what to do based on current state could be random based on epsilon
        action = agent.get_action(obs)
        # take the action with env.step(action) and then recieve the results
        next_obs, reward, terminated, truncated, info = env.step(action)
        # update the agent based on the q values
        # print("updatingq-values...")
        agent.update(obs, action, reward, terminated, next_obs)
        # update the current state observation
        obs = next_obs
        # check if the game is terminated or truncated meaning out of time
        done = terminated or truncated 
    
    agent.decay_epsilon()

print(len(agent.training_error))
# training plot visuals 
def get_moving_avgs(arr, window, mode="valid"):
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=mode) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
axs[0].plot(get_moving_avgs(env.return_queue, rolling_length, "valid"))

axs[1].set_title("Episode lengths")
axs[1].plot(get_moving_avgs(env.length_queue, rolling_length, "valid"))

axs[2].set_title("Training Error")
axs[2].plot(get_moving_avgs(agent.training_error, rolling_length, "same"))

plt.tight_layout()
plt.show()