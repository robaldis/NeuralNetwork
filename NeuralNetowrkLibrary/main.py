import neuralNetwork as nn
from numpy import *


def main():

    input = array([1,0])

    brain = nn.NeuralNetwrok(2,2,1)
    output = brain.feedForward(input)
    print (output)



if (__name__ == "__main__"):
    main()
