# this line imports something called default dict which if you try to access a key in a dictionary it creates it automatically using the function as the value
from collections import defaultdict
import gymnasium as gym
# numpy is fundamental for machine learning because it allows you to do a wide range of dsa and math more effeciently and effectively
import numpy as np

# we make a class so we can encapsulate our agent so it is reusable remembers everything specific to that instance and keep organized
class CartpoleAgent:
    def __innit__(
            self,
            # defines a standardized structure for environments to work with reinforcement learning agents includes methods like reset() and action_space
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            # remember this is gamma and how much it cares about future rewards
            discount_factor: float = 0.95
    ):