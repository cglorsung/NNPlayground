# Author : Conor Lorsung following the examples found at https://iamtrask.github.io/2015/07/12/basic-python-network/
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import numpy as np


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
                   [0, 1, 1] ])

hiddenArr = np.array([ [1],
                       [.75],
                       [.5],
                       [.25],
                       [1] ])

np.random.seed(10)

synapse0 = 2*np.random.random((3,4))-1
synapse1 = 2*np.random.random((4,1))-1

for i in range(1000):
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
print('LAYER 0:\n', lay0, '\nLAYER 1:\n', lay1, '\nLAYER 2:\n', lay2)

