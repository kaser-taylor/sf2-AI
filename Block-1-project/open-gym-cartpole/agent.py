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
        
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        self.env = env
        # so in the gymnasium environmtne there is a q-value table and we reference it with self.q_values
        # default dict creates a key for the values
        # an array with the zeroes to the number of actions
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []


    # so obs is the observation of the game state its usually a 1D numpy array that looks like this
    # obs = np.array([position, velocity, angle, angular_velocity])
    # we pass the observation in as the argument
    def get_action(self, obs: np.ndarray) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

        # this is how epsilon greedy is implemented. if a random number is less than the epsilon you sample the action space and explore if it is greater you choose the network action

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[obs]))
