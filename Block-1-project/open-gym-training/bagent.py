from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        # this is alpha basically how much corrections matter
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        # discount factor is gamma how much it cares about future rewards
        discount_factor: float = 0.95):

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
        # this line of code  is new to me gonna give a gpt description
        # self.q_values is the q table in the class and its gonna store learned values in a pair similar to q_values[state][action] = score
        # uses something called default dict which is from the collections module, if you try to access a key that doesn't exist it creates it automatically using the function you give it
        # lambda is like a one liner function
        # env.action_space.n is the total number of possible actions ex cartpole/flappy bird there are two but in black jack there are a 2 maybe idk black jack that will
        # np.zeros() creates a numpy array for the new state
        # so the jist of this is that every new state gets a fresh array of q values 
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []


    # this looks brand new too expl incoming
    # the purpose of this function is to choose an action based on the current state of the env
    # so obs is a tuple of values that contains an action spaces to store in a q table that are normalized
    def get_action(self, obs: tuple[int, int, bool]) -> int:
            """
            Returns the best action with probability (1 - epsilon)
            otherwise a random action with probability epsilon to ensure exploration.
            """
            # with probability epsilon return a random action to explore the environment

            # this implements the epsilon greedy exploration if the float generated by np is less than the epsilon value you explor otherwise you use the known q table action. so as epsilon decays random actions become less likely 
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            
            else:
                return int(np.argmax(self.q_values[obs]))

    # ive never seen this many arguments lol
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float, 
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ): 
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) *np.max(self.q_values[next_obs])
        # this looks like the bellman ford equation
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        # this looks similar to the mean squared error but not squared maybe less severe learning is the resule
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
