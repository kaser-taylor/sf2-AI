Video resource - https://www.youtube.com/watch?v=aircAruvnKk

Notes: 

- Plain / one of the first Neural networks is called a multilayer percpetron. It was used to identify numbers that were handrwarn in a resolution of 28x28 px. 

- Neurons is a thing that holds a number between 0 and 1 

- A perceptron will have a neuron for each of the pixels 28x28 so 784 neurons holds a number that represents the grayscale value of the pixel
    - each number is referred to as its activation
    - lights up when high number
    - 784 is first layer

- last layer has 10 neurons one for each digit
    - represents the likelihood of a given input being a given number

- In the middle are hidden layers which do the magic math 

- Why is it reasonable for layers to behave intelligently 
    - Humans chunck fragments of an image
    - we hope in the second to last layer the nerual net is doing the same thing
    - last layer is which combo in second to last = a number
    - basically the information is chunked out maybe? we might not know kind of black box
    - can do all sorts of things that are not just images
        - through chunking

- how do activations work?
    - use weights to connect neuron layers
    - take all activations from the first layer and compute their sum according to the weights 
    - then you pump weighted sum into a function that squishes the activation into some number between 0 and 1
        - common function that does this is called a sigmoid function or logistic curve 
        - kinda looks like it has limits at y = 0 and y = 1 so very positive are very close to 1 and very negative are very close to 0
    - so an activation(weights plus first layer activation) is basically how positive the weighted sum is 
    - you add bias for inactivity
        - what this means is you only want the neuron to activate if it is greater than a certain number. this basically equates like what an activation potential is in chemistry or for human neurons so that way things just aren't firing with minimal input
        - to add the bias you add it to the weighted sum before the sigmoid squishification
            - in the video he uses negative 10
    - all this happens for one neuron in the second layer and each neuron is connected to each 784 pixel neurons. this means a ton of calculations
        - each connection has its own weight and bias
        - total of 784 * 16 * 16 so 13000 in total for whole net

- more accurate to think of each neuron as a function that takes in all activations of previous neurons and spits out a number between 0 and 1
