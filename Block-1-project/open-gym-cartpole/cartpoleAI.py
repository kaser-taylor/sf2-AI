import gymnasium as gym
import pickle
import numpy as np
from collections import defaultdict

with open("trained_cart", "rb") as f:
    q_table_data = pickle.load(f)



q_values = defaultdict(lambda: np.zeros(2))  # 2 actions in CartPole
q_values.update(q_table_data)

env = gym.make("CartPole-v1", render_mode="human")

def discretize(obs: np.ndarray) -> tuple:

    obs = np.clip(obs, [-4.8, -5, -0.418, -5], [4.8, 5, 0.418, 5])

    # define a bin for each state dimension eg velocity x y etc
    bins = [
        # the lin space values aren't empty because youre supposed to put ranges on what you expect the values to be
        # lin space returns evenly spaced values between the start , the end, and the number of values it returns is the last number 
        np.linspace(-4.8, 4, 5),
        np.linspace(-5, 5, 5),
        np.linspace(-0.418, 0.418, 5),
        np.linspace(-5, 5, 5),
    ]

    # an empty list to store the values
    discretized = []

    # for i in range number of observation state vector things (velocity x y etc)
    for i in range(len(obs)):
        # turns the obs number into the closest bin number and appends it into our discretized value list
        discretized.append(np.digitize(obs[i], bins[i]))
    
    # returns as tuple for our update function
    return tuple(discretized)

for episode in range(2):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        state = discretize(obs)
        action = np.argmax(q_values[state])  # pure exploitation
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Episode {episode + 1} finished in {steps} steps with reward {total_reward}")

env.close()