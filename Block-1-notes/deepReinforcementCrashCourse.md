Video Resource: https://www.youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi 

Notes: 
- Video #1 
    - gymnasium is a library of reinforcement library environments
        - he used it for flappy bird
    - So this video is gonna be super similar to what im trying to do cause rather than a computer vision projet is uses the memory values for various game components 
    - maybe research package managers and dev environments this might be needed doesn't seem nessecary (idk if i spelled that right) for the kind of work ive done so far
    - he says not to train a complicated environment first 

- Video #2
    - implement dqn with pytorch
    - in a dqn the input layer represents the current state
    - ouput layer represents available actions
        - so in my street fighter network i was wondering if a simple dqn would learn move chaining and it sounds like it won't and I will have to use reward shaping to add it. It sounds like the memory of the game does store something like combo counters
        - the following bullet should be used for when i get to introducing combos 
            - 1. Temporal Credit Assignment
                If your reward comes only after the full combo lands, the DQN must learn that the sequence (e.g. → ↓ ↘ + punch) leads to a high reward.

                Through training, it slowly starts assigning value to earlier parts of the combo.

                🧠 It’s like learning that "crouch" is good because it leads into a fireball.

                2. Experience Replay + Target Networks
                These help stabilize training so that sequences of actions aren’t lost in noise.

                You need to make sure you’re sampling enough good sequences so it connects the dots.

                3. Frame-Skipping / Action-Repeats
                Often used in Atari environments—this helps the agent repeat actions across multiple frames.

                Useful for pressing buttons long enough to execute a move.
            - you can make new networks and load old weights into them which may be useful
    
    - q values or quality is the reward that is expected to be returned for taking an action
    - network that is trained is called the policy network
    - hidden layers vary based on complexity
    - so he has two files an agent and a network. the agent takes action, updates the network, and plays the game 
    - the network is the brain that decides the actions
    - use init in pytorch to define the layers
    - use the forward function to do the math
    - 

    - so this is a code example from the flappy bird game and I'm struggling with his explanation so im gonna do some gpt 
     - if --name == "main":
        state_dim = 12
        action_dim = 2
        net = DQN(state_dim, action_dim)
        state = torch.randn(10, state_dim)
        output = net(state)
        print(output)
        - so he said that input layer values are implicit and i guess they are defined by state_dim. in my streetfighter game this would be all they like x y values of the player, combo state, etc...
        - so action_dim are the number of actions the agent file can choose. 
        - in sf2 this is gonna be initially for the first network things like kick punch jump forward backward etx
        - the net = DQN(state_dim, action_dim) passes the state and action into a dqn class that runs the network calcs
        - state = line creates fake data to test if the network actually runs and works 
        - output = tests the network with the state values

    - so he goes back to the agen file and he wraps in in a class and the code in a run function

    - imports dqn 

- Video #3 
     - So he is talking about something called Experience Replay
        - experience replay consists of state, action, new_state, reward, terminated and is passed into a deque
            - remember a deque is a double ended que that allows you to pop and add to both ends of the queue in o1 time
        - so it uses this dequeu so the higher reward experiences are closer to the front and the old ones get popped off the back so you dont run out of memory
        - he then creates a new experience replay file
        - heres some gpt experience replay notes 
            - these are the things we need to import for the class from collections 
            import deque
            import random
            import numpy as np
            import torch
            - next we need to decide the max number of states we want to store in the deque and we do this with the following 
            - class ReplayBuffer:
            def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
            - next we need a push method that adds the experience which is one time step to the bufffer 
            -     def push(self, state, action, reward, next_state, done):
            experience = (state, action, reward, next_state, done)
            self.buffer.append(experience)

        - so now hes adding something called replay memory
            - so this basically creates a specified number of the replay class that acts in a dequeue fashion
            - im guessing this is a built in class in pytorch
        
        - so he does something interesting. in training you train for a specified number of episodes in pytorch but he imports this tool called itertools which allows you to generate a number indefinetely and then he will stop it manually. I could see using this having its pros and its cons. its pro is definetely you can run it till your satisfied but if youre using resources that are too expensive and forget to shut it off this could be very bad

        - so he just mentioned something called the epsilon greedy algorithm. Idk what that is and I don't think he is going to go in depth so im gonna do some gptsearch
            - The ε-greedy algorithm says:
                With probability ε → take a random action (explore)
                With probability 1 - ε → take the best action from your Q-network (exploit)
            - so It sounds like you start with a high epsilon value and if that epsilon value is close to one it takes a random action. this is important cause it keeps the dqn from abusing one game action like punching a million times in a row. and you code in a decay so as the training iterates it takes the action that delivers the highest reward
                - i imagine if you have the resources its viable to make the decay rate and training iterations very high for iterations and low for decay rate so it gets to explore a lot first. theres obviously gonna be a law of diminishing returns on this but if you have the resources why not
        
        - so now hes getting into hyperparameters
            - a hyperparameter is a tool that you control that the ai does not learn
            - an example is learning rate epsilon hidden layer size etc.
            - 

- Video #4
    - Implement Epsilon Greedy and Debug training loop

    - so this seems like a lot of math implementation using built in functions im not particularly sure how to take notes on this video 

    - he keeps bringing up tensors and idk what that is
        - a tensor is a data structure that is basically nested lists of numbers a 0d tensor is something like an integer no list a 1d tensor is a list of integers a 2d tensor is a list of lists like a matrix and a 3d tensor is a stack of matrixes like an rgb image 


- Video #5
    - this video is about implementing a target network
    - Okay so I was super confused on target networks and this is the jist. If you have your primary network recursively correct itself based on reward its similar to the event replay problem and it might choose short term reward and suboptimal strategies.
    
     so you basically have an older sub optimal network that the network compares what the actual reward was to the older predicted award
        - this sentence is wrong and it is actually based on something called the Bellman Equation
        - based on this equation 
            - target_Q = reward + γ * max(Q_target(next_state))
        - so reward is the reward that the agent just got for taking an action in the current state
        - gamma is the discount factor which is how much it cares about future rewards
        - and then max q_target next state
            - the target network predicting the q-values for the next state and gives you a future value estimat
            - this is the one that confuses me the most 
                - so this value is what the target network says the maximum reward for playing the rest of the game perfectly is
        
        - so the reason we use target Q after all this math 
            - it is the value the q netowrk should have predicted so you try to train the q network to match this 

            - remember this is about incrementally getting better over time not a oneshot best ever training session
        - heres a history of the bellman equation from gpt for more context 
            - Instead of trying to solve the full decision path in one go (which explodes in complexity), he said:

            “What if I solve it backwards by defining the value of a state in terms of the value of the next state?”

            This idea is now known as Bellman’s Principle of Optimality:

            "An optimal policy has the property that, whatever the initial state and decision, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."

            Basically: if you’re halfway through your plan, the rest of it should still be optimal from that point forward.

            🧮 And Thus… The Bellman Equation Was Born
            The original Bellman equation was written like this:

            text
            Copy code
            V(s) = max_a [ R(s, a) + γ * V(s') ]
            Where:

            V(s) = value of being in state s

            R(s, a) = reward for taking action a in state s

            s' = the next state

            γ = discount factor (how much we care about future rewards)

            It recursively defines the value of a state based on the best thing you can do next.

            This idea became the core of dynamic programming and later, reinforcement learning.

     and then you store the loss and train based on that.

     - what is an episode?
        - in rl an episode is one complete run through the environment from the starting state to an end state like the game over a win a loss or a timeout 
    
        - reminder an experience is a choice that the network makes along with the current state
    
    - q learning algo Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        - Q(s, a)
            - update the q value for this state, action pair
        - alpha (the learning rate)
            - if alpha = 1 overwrite it completely if alpha = 0 don't learn at all. it controls how fast you learn
        - r + gamma 
            - reward for the current action + gamma the how much we care factor 
        - last bit is best furture option and best possible reward bellman
    - the whole term is temporal difference error
        - basically if its big update a lot
        - if its small its close 
        - I haven't read much about alpha im gonna take some more notes 
        - so alpha is how seriously we take our error 
    
    - after coming back to this I am confused about the difference between alpha and gamma
        - alpha is how big the updates are
        - gamma is how much future rewards matter
    
    - so hes defining loss and optimization but I believe those are topics for next video

    - quick notes on what optimization is
        - an optimizer is an algorithm that updates the models weights
        - 


- Video #6
    - okay so this is a 4 min video and targets what seem like really important concepts to know so im gonn have a lot of gpt research notes for this video

    - so loss is = to mean square error (curent q, and target q)
         - question is what is loss, and what is mean square error
         - loss 
            - loss is a number that tells the model how wrong it is
                - the difference btween what it predicted vs what the correct answer was
            - so mean square error is basically the ((reward + gamma + bellman ford q value) - (q of the prediction))**2
                - this is important because it takes the actual reward from the environment into account and stabilizes it using the bellman equation and gamma and then punishes hard for big errors and little for small errors cause of the 2
                - I was confused cause what does it mean if the prediction value is "better"/ higher than the bellman target network and it still gets punished because it's basically overconfident in the wrong action. rl is an iterative and incremental approach not a one shot learning run.
            - so alpha is learning rate and this is super important!!! i really like the graph he did in this video
                - since mean squared error is a quadratic equation is is a parabola
                - you have minimized loss when you get to the vertex.
                - if alpha is too high you will just bounce off each side of the parabola and you won't be able to optimize the loss and get it to 0 cause you overcorrect the weight every time
                - small steps are better if you have resources obviously if you can be optimal be optimal
            - 

- Video #7 pytorch code
    - this video seems like a learn by doing once i get there sort of deal maybe well review it for this blocks mini project

- Video #8 testing dqn
    - I think the rest of these videos are going to be more practical and less note taking on concepts 

- Video #9 train on flappy bird
    - 

- video #10 double dqn
    - stopped watching here seems out of scope for this block
    

