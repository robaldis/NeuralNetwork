from numpy import *
from random import *
import math


def activation(x):
    # Sigmoid
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)


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

        # Add up the hidden weights with the inputs
        hidden = dot(self.weights_ih, input)
        # Adding the bias for the hidden layer
        hidden = hidden + self.bias_h
        hidden = matrix.transpose(hidden)

        # Getting the activation value for the hidden layer
        for i in range((size(hidden, 0))):
            for j in range(size(hidden, 1)):
                hidden[i][j] = activation(hidden[i][j])

        # Adding up the weights with the hidden layer
        output = dot(self.weights_ho, hidden)
        output = matrix.transpose(output)
        # Adding the bias for the output layer
        output = output + self.bias_o

        # Getting the activation value for the output layer
        for i in range(size(output,0)):
            for j in range(size(output, 1)):
                output[i][j] = activation(output[i][j])

        # Sending it back to the caller
        return output


    def randomWeights(self, n):
        for i in range(size(n, 0)):
            for j in range(size(n, 1)):
                n[i][j] = uniform(-1,1)
        return n
