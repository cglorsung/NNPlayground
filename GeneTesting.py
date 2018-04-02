# Author : Conor Lorsung
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import numpy as np

# Supervised?
sup = True

# Number of iterations
iterations = 10000

# Initialize data array
datArr = []

# Class label array
classArr = []

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


if sup:
    outArr = np.array([[1], [1], [1], [1], [0], [0], [0], [0], [0], [1], [0], [1]])
else:
    # Initialize hypotheses at 0
    outArr = np.array([[0]] * len(datArr))

# Set random seed
np.random.seed(1)

# Evaluate synapses
synapse0 = 2*np.random.random((len(datArr[0]), len(datArr))) - 1
synapse1 = 2*np.random.random((len(outArr), len(outArr[0]))) - 1

# Run the system
for i in range(iterations):
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

datArr = []

# Read data file
file = open('Data\\AllGenes.csv', 'r')

# Data file is type CSV, so split on comma's by line
for n in file.readlines():
    datArr.append(n.split(','))

for n in range(0, len(datArr)):
    classArr.append(datArr[n][len(datArr[n])-1])
    datArr[n] = datArr[n][:-1]

# Convert datArr into numpyArray(float) type
datArr = np.array(datArr).astype(float)

# Linear normalization of the data-set
datArr = datArr / np.linalg.norm(datArr)

labALL = []
labAML = []

for n in range(0, len(datArr)):
    lay0 = datArr[n]
    lay1 = sig(np.dot(lay0, synapse0))
    lay2 = sig(np.dot(lay1, synapse1))
    if classArr[n].__contains__('ALL'):
        labALL.append(lay2)
    else:
        print(datArr[n])
        labAML.append(lay2)

print('\n')
print('AML')
for n in labAML:
    print(n[0])

print('\n')
print('ALL')
for n in labALL:
    print(n[0])