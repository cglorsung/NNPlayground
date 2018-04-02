# Author : Conor Lorsung
# Purpose: This project is simply a neural network playground
#          for me to learn more about building neural networks,
#          their implementations, and their strengths/weaknesses

import random
import numpy as np


# This function generates arrays of arrays
# Variables:
#   outerLength : Number of inner arrays to be generated
#   innerLength : Number of values to be put into each inner array
#   minimum     : The minimum value for the random generator
#   maximum     : The maximum value for the random generator
#   integers    : Boolean for generating integers vs. floats
def genRandArrays(outerLength, innerLength, minimum, maximum, integers):
    finArr = []
    for i in range(outerLength):
        arr = []
        for j in range(innerLength):
            if integers:
                arr.append(random.randint(minimum, maximum))
            else:
                arr.append(random.uniform(minimum, maximum))
        finArr.append(arr)
    return np.array(finArr)


print(genRandArrays(5, 3, 0, 1, True))