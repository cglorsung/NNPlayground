# Author : Conor Lorsung following the examples found at https://iamtrask.github.io/2015/07/12/basic-python-network/
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import numpy as np
import Utilities.GenArrays as ga


# Sigmoid function (logistic)
def sig(x, derive=False):
    if derive:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


inArr = np.array([ [1, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 0] ])

hiddenArr = np.array([ [1],
                       [0],
                       [0],
                       [0],
                       [1],
                       [0],
                       [0],
                       [1] ])

np.random.seed(10)

synapse0 = 2*np.random.random((3,4))-1
synapse1 = 2*np.random.random((4,1))-1

for i in range(10000):
    lay0 = inArr
    lay1 = sig(np.dot(lay0, synapse0))
    lay2 = sig(np.dot(lay1, synapse1))

    error2 = hiddenArr - lay2
    delta2 = error2 * sig(lay2, derive=True)

    if i%1000 == 0:
        print(i, ': Error: ', np.mean(np.abs(error2)))

    error1 = delta2.dot(synapse1.T)
    delta1 = error1 * sig(lay1, derive=True)

    synapse1 += lay1.T.dot(delta2)
    synapse0 += lay0.T.dot(delta1)


print('Complete')
print('LAYER 2:\n', lay2)

# Set up the training array
tester = ga.genRandArrays(10, 3, 0, 1, True)

# Run the NN with testing data
for i in tester:
    lay0 = i
    lay1 = sig(np.dot(lay0, synapse0))
    lay2 = sig(np.dot(lay1, synapse1))
    if lay2 > .9:
        print('PASS : ', i, ' : ', lay2)
    else:
        print('FAIL : ', i, ' : ', lay2)

