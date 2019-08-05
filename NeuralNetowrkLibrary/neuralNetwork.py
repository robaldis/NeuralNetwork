from numpy import *
from random import *
import math
import scipy.special


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

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def feedForward(self, inputArray):
        # convert inputs list to 2d array
        input = array(inputArray, ndmin=2).T

        # Calculate singnals into the hidden layer
        hiddenInput = dot(self.weights_ih, input)
        hiddenInput = hiddenInput + self.bias_h
        hiddenOutput = self.activation_function(hiddenInput)
        # hiddenOutput = matrix.transpose(hiddenOutput)

        # Calclate signals into the final layer
        finalInput = dot(self.weights_ho, hiddenOutput)
        finalInput = finalInput + self.bias_o
        finalOutput = self.activation_function(finalInput)

        # Sending it back to the caller
        return finalOutput


    def randomWeights(self, n):
        for i in range(size(n, 0)):
            for j in range(size(n, 1)):
                n[i][j] = uniform(-1,1)
        return n
