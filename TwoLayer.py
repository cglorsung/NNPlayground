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
                   [0, 1, 0],
                   [1, 0, 0],
                   [0, 1, 1] ])

ouArr = np.array([[1, 0, 1, 0]]).T

np.random.seed(10)

synapse = 2*np.random.random((3,1))-1

for i in range(10000):
    lay0 = inArr
    lay1 = sig(np.dot(lay0, synapse))

    error = ouArr - lay1

    delta = error * sig(lay1, True)

    synapse += np.dot(lay0.T, delta)


print('Complete')
print(lay1)