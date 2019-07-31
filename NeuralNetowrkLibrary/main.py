import neuralNetwork as nn
from numpy import *


def main():

    input = array([1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1])

    brain = nn.NeuralNetwrok(20,100,2)
    output = brain.feedForward(input)
    print (output)



if (__name__ == "__main__"):
    main()
