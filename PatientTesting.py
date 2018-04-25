# Author : Conor Lorsung
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import numpy as np

# Supervised?
sup = True

# Number of iterations
iterations = 10000000

# Initialize data array
datArr = []

# Class label array
classArr = []

# Read data file
file = open('Data\\SampleSet.csv', 'r')

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
    parts = []
    outArr = np.array([[]])
    for i in range(0, 100):
        parts.append([0])
    for i in range(0, 100):
        parts.append([1])
    # outArr = np.array([[1], [1], [1], [1], [0], [0], [0], [0], [0], [1], [0], [1]])
    outArr = np.array(parts)
else:
    # Initialize hypotheses at 0
    outArr = np.array([[0]] * len(datArr))

# Set random seed
np.random.seed(1)

# Evaluate synapses
synapse0 = 2*np.random.random((len(datArr[0]), len(datArr))) - 1
print('syn0 ', synapse0)
synapse1 = 2*np.random.random((len(outArr), len(outArr[0]))) - 1
print('syn1 ', synapse1)
synapse2 = 2*np.random.random((len(synapse1[0]), len(synapse1))) - 1
print('syn2 ', synapse2)
synapse3 = 2*np.random.random((len(outArr), len(outArr[0]))) - 1
print('syn3 ', synapse3)

# Run the system
for i in range(iterations):
    lay0 = datArr
    lay1 = sig(np.dot(lay0, synapse0))
    lay2 = sig(np.dot(lay1, synapse1))
    lay3 = sig(np.dot(lay2, synapse2))
    lay4 = sig(np.dot(lay3, synapse3))

    error4 = outArr - lay4
    delta4 = error4 * sig(lay4, derive=True)

    error3 = delta4.dot(synapse3.T)
    delta3 = error3 * sig(lay3, derive=True)

    error2 = delta3.dot(synapse2.T)
    delta2 = error2 * sig(lay2, derive=True)

    error1 = delta2.dot(synapse1.T)
    delta1 = error1 * sig(lay1, derive=True)

    synapse3 += lay3.T.dot(delta4)
    synapse2 += lay2.T.dot(delta3)
    synapse1 += lay1.T.dot(delta2)
    synapse0 += lay0.T.dot(delta1)

# Indicate completion
print('Complete')

# Output the final (hypothesis) layer of the NN
print(lay4)
print(len(lay4))
'''
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
    lay3 = sig(np.dot(lay2, synapse2))
    lay4 = sig(np.dot(lay3, synapse3))
    if classArr[n].__contains__('ALL'):
        labALL.append(lay4)
    else:
        print(datArr[n])
        labAML.append(lay4)

print('\n')
print('AML')
for n in labAML:
    print(n[0])

print('------------------')

print('\n')
print('ALL')
for n in labALL:
    print(n[0])
'''
