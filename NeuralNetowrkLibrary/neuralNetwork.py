from numpy import *
from random import *
import math


def activation(x):
    # Sigmoid
    return 1/(1 + math.exp(-x))

def matrixAdd(n, b):
    token = list()
    for i in range(size(b,0)):
        n[i] += b[i]
    return n



class NeuralNetwrok (object):

    def __init__(self,input_nodes,hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Make the weights between the input and hidden and hidden and output
        self.weights_ih = ndarray(shape = (self.hidden_nodes, self.input_nodes))
        self.weights_ho = ndarray(shape = (self.output_nodes, self.hidden_nodes))
        # randomise the weights
        self.weights_ih = self.randomWeights(self.weights_ih)
        self.weights_ho = self.randomWeights(self.weights_ho)

        self.bias_h = ndarray(shape = (self.hidden_nodes,1))
        self.bias_o = ndarray(shape = (self.output_nodes, 1))
        self.bias_h = self.randomWeights(self.bias_h)
        self.bias_o = self.randomWeights(self.bias_o)
        self.bias_h = matrix.transpose(self.bias_h)
        self.bias_o = matrix.transpose(self.bias_o)



    def feedForward(self, input):

        hidden = dot(self.weights_ih, input)
        hidden = hidden + self.bias_h
        # hidden = matrixAdd(hidden, self.bias_h)
        # Getting the activation value for the hidden layer
        for i in range((size(hidden, 0) - 1)):
                hidden[i] = activation(hidden[i])

        hidden = matrix.transpose(hidden)
        output = dot(self.weights_ho, hidden)
        output = output + self.bias_o
        # Getting the activation value for the output layer
        for i in range((size(output,0))):
                output[i] = activation(output[i])

        # Sending it back to the caller
        return output

    def randomWeights(self, n):
        for i in range(size(n, 0)):
            for j in range(size(n, 1)):
                n[i][j] = uniform(-1,1)
        return n
