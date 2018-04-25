# Author : Conor Lorsung
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import numpy as np


# Number of iterations of training
iterations = 1000

# Logistic sigmoid function
def sig(x, derive=False):
    if derive:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


# Input array
inArr = np.array([ [0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1] ])

# Output (expected values) array
ouArr = np.array([ [0],
                   [1],
                   [1],
                   [0] ])

# Set the seed for randomizer
np.random.seed(52)

# Dimension variables
inLen0 = len(inArr)
inLen1 = len(inArr[0])
ouLen0 = len(ouArr)
ouLen1 = len(ouArr[0])

# Synapses
sy0 = 2 * np.random.random((inLen1,inLen0)) - 1
sy1 = 2 * np.random.random((ouLen0,ouLen1)) - 1

# Run it
for i in range(iterations):
    lay0 = inArr
    lay1 = sig(np.dot(lay0, sy0))
    lay2 = sig(np.dot(lay1, sy1))

    error2 = ouArr - lay2
    delta2 = error2 * sig(lay2, derive=True)

    error1 = delta2.dot(sy1.T)
    delta1 = error1 * sig(lay1, derive=True)

    sy1 += lay1.T.dot(delta2)
    sy0 += lay0.T.dot(delta1)


print('Complete')
print('Final layer:\n', lay2)
print('\n')

# Test value array
tester = np.array([ [1,0], [0,1], [1,0], [0,1], [1,1], [0,0], [1,1], [0,0] ])

for i in tester:
    lay0 = i
    lay1 = sig(np.dot(lay0, sy0))
    lay2 = sig(np.dot(lay1, sy1))

    # Where's my ternary at?
    if lay2 > .5:
        print('TRUE : ', i, ' : ', lay2)
    else:
        print('FALSE: ', i, ' : ', lay2)