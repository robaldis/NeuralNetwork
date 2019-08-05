import neuralNetwork2 as nn
from numpy import *


def main():

    input = [1,0,1,0]

    brain = nn.NeuralNetwrok(4,5,6)
    output = brain.feedForward(input)
    print (output)



if (__name__ == "__main__"):
    main()
