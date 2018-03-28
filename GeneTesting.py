# Author : Conor Lorsung
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import numpy as np

# Initialize data array
datArr = []

# Read data file
file = open('Data\\SampleGenes.csv', 'r')

# Data file is type CSV, so split on comma's by line
for n in file.readlines():
    datArr.append(n.split(','))

# Convert datArr into numpyArray(float) type
datArr = np.array(datArr).astype(float)

# Linear normalization of the data-set
datArr = datArr / np.linalg.norm(datArr)


# Sigmoid function (logistic)
def sig(x, derive=False):
    if derive:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


outArr = np.array([ [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0] ])

# Set random seed
np.random.seed(1)

# Evaluate synapses
synapse0 = 2*np.random.random((len(datArr[0]), len(datArr))) - 1
synapse1 = 2*np.random.random((len(outArr), len(outArr[0]))) - 1

# Run the system
for i in range(40000):
    lay0 = datArr
    lay1 = sig(np.dot(lay0, synapse0))
    lay2 = sig(np.dot(lay1, synapse1))

    error2 = outArr - lay2
    delta2 = error2 * sig(lay2, derive=True)

    error1 = delta2.dot(synapse1.T)
    delta1 = error1 * sig(lay1, derive=True)

    synapse1 += lay1.T.dot(delta2)
    synapse0 += lay0.T.dot(delta1)

# Indicate completion
print('Complete')

# Output the final (hypothesis) layer of the NN
print(lay2)