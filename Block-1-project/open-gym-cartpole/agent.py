# this line imports something called default dict which if you try to access a key in a dictionary it creates it automatically using the function as the value
from collections import defaultdict
import gymnasium as gym
# numpy is fundamental for machine learning because it allows you to do a wide range of dsa and math more effeciently and effectively
import numpy as np

# we make a class so we can encapsulate our agent so it is reusable remembers everything specific to that instance and keep organized
class CartpoleAgent:
    def __init__(
            self,
            # defines a standardized structure for environments to work with reinforcement learning agents includes methods like reset() and action_space
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            # remember this is gamma and how much it cares about future rewards
            discount_factor: float = 0.9999
            
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
            state = self.discretize(obs)
            return int(np.argmax(self.q_values[state]))
    
    # since game state is continuous there are infinite possible values and the neural net needs discrete values to pull from and run. so this function puts those values into a fixed number of discrete bins for the update function to pull from and they end up becoming tuples which can be hashed indexed and 
    def discretize(self, obs: np.ndarray) -> tuple:

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
    
    # I copied this update function from black jack. GPT says that in cartpole np arrays are not hashable and cannot be dictionary keys so we have to do something called discretizing which i know nothing about but I'm gonna follow its line of thinking and take some notes. 
    def update(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: np.ndarray
    ):
        """Updates the Q-value of an action."""
        # gotta add those discrete values
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        future_q_values = (not terminated) * np.max(self.q_values[next_state])


        temporal_difference = (
            reward + self.discount_factor * future_q_values - self.q_values[state][action]
        )

        # temporal_difference = (
        #     reward + self.discount_factor * ((future_q_values - self.q_values[state][action]) ** 2)
        # )

        if abs(temporal_difference) > 1e-5:
            self.q_values[state][action] += self.lr * temporal_difference
        self.training_error.append(abs(temporal_difference))

    
    def decay_epsilon(self):
        # self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        self.epsilon = max(self.final_epsilon, self.epsilon * .995)
