Video Resource: https://www.youtube.com/watch?v=JgvyzIkgxF0

Notes: 

- Supervised Learning
    - You give a neural net an input and you know what it should output use an algorithm to train the neural net to produce that output

- he keeps bringing up back propagation idk what that is look up later

- so a neural net that produces an action from input frames in the context of reinforcement learning is called a policy network and they are trained using something called policy gradients

- a policy gradient is you start with a completely random network
    - use something called sampling to try different probabilities of your output action
    - you then use rewards often a scoreboard or something positive.
    - the network then optimizes itself to try and earn as much reward as possible
        - humans are kind of like this and i can see an example of getting stuck in loops of short term gain but long term harm as something like smoking cigarettes maybe this is why you use sampling so you dont get stuck 
    - so in a roundabout term since you know what policy led to a point you can use a function to increase the probability of that action happening in the future that ended in a reward. and use in a roundabout way to decrease policies that ended in a loss
        - why quitting smoking is so hard for humans
        - ex chantix adds negative reward for smoking

- there are downsides to using this method
    - policy gradients can assume a whole branch of actions are bad even though it might just be one
        - called credit assignment problem
        - happens cause of sparse reward 
        - very sample inneffecient
        - increase or change rewards? 
        - So he talks about how when there is a complex series of actions it is really hard to train a sequence of events because there are too many that are too random due to sampling that the net may never recieve an award. this could be a problem in sf2 

    - this is why supervised deep learning is so successful because it allows you to effeciently train a model but could never be better than a human

    -  to solve this issue is called reward shaping
        - solved montezumas revenge by giving extra rewards!!
        - useful for sf2
        - downsides needs to be redone for every new environment. 
        - too many new reward functions
        - suffers from alignment problem
        - will find way to get a lot of reward but not do what you want to do
            - not general
        - this will be a problem for sf2
        - 
