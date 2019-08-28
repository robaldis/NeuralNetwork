import numpy as np
import random


layer_sizes = (2,3,3,6)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class NeuralNetwork():

    def __init__(self, layer_sizes):
        np.random.seed(1)
        random.seed(1)
        weight_shape = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shape]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]

        self.weights_output = []#[np.zeros((s,1)) for s in layer_sizes[1:]]


        self.lr = 0.1

        self.num_layers = len(layer_sizes)

        self.activation_function = lambda x : 1/(1+np.exp(-x))


    def predict(self, a, target = None, training = False):
        self.weights_output.append(a)
        i = 0
        for w,b in zip(self.weights, self.biases):
            #self.weights_output.append(a)
            a = self.activation_function(np.matmul(w,a) + b)
            self.weights_output.append(a)
            
            i += 1

        #for i in self.weights_output:
        #    print (f"{i}\nOUTPUT\n")
        return a

    def train(self,inputArray , targetArray):

        self.predict(inputArray)

        #weights_ih = self.weights[0]
        #weights_ho = self.weights[1]

        #bias_o = self.biases[1]
        #bias_h = self.biases[0]

        #finalOutputs = self.weights_output[1]
        #hiddenOutput = self.weights_output[0]

        
        #target_T = np.array(targetArray, ndmin=2).T

        ## Calculate the errors for the output layer
        #outputErrors = target - finalOutputs
        #outputGradiants = finalOutputs * (1- finalOutputs)
        #outputGradiants = outputGradiants * outputErrors
        #outputGradiants = outputGradiants * self.lr

        #hidden_T = np.matrix.transpose(hiddenOutput)

        #weights_ho += outputGradiants * hiddenOutput
        #bias_o += outputGradiants

        ## Calculate the hidden layer errors
        #who_T = np.matrix.transpose(weights_ho)
        #print (weights_ho)
        #print (outputErrors)
        #hiddenErrors = who_T * outputErrors

        ## Hidden layer gradiant
        #hiddenGradiants = hiddenOutputs * ( 1 - hiddenOutput)
        #hiddenGradiants = hiddenGradiants * hiddenErrors
        #hiddenGradiants = hiddenGradiants * self.lr

        ## Calculate deltas
        #input_t = np.matrix.transpose(target)
        #weights_ih_deltas = hiddenGradiants * input_t

        ## change the weights by the deltas
        #self.weights_ih += weights_ih_deltas
        ## Change the bias with the gradiant
        #self.bias_h += hiddenGradiants



        #-----------------------------------------------



        ## convert inputs list to 2d array
        #input = np.array(inputArray, ndmin=2).T

        ## Calculate singnals into the hidden layer
        #hiddenInput = np.dot(weights_ih, input)
        #hiddenInput = hiddenInput + bias_h
        #hiddenOutput = self.activation_function(hiddenInput)
        ## hiddenOutput = matrix.transpose(hiddenOutput)

        ## Calclate signals into the final layer
        #finalInputs = np.dot(weights_ho, hiddenOutput)
        #finalInputs = finalInputs + bias_o
        #finalOutputs = self.activation_function(finalInputs)

        #target = np.array(targetArray, ndmin=2).T

        ## Calculate the errors for the output layer
        #outputErrors = target - finalOutputs
        #outputGradiants = finalOutputs * (1- finalOutputs)
        #outputGradiants = outputGradiants * outputErrors
        #outputGradiants = outputGradiants * self.lr

        ## Calclate deltas
        #hidden_T = np.matrix.transpose(hiddenOutput)
        #weights_ho_deltas = outputGradiants * hidden_T

        ## Change the weights by the deltas
        #weights_ho += weights_ho_deltas
        ## Change the bias with the gradiant
        #bias_o += outputGradiants

        ## Calculate the hidden layer errors
        #who_T = np.matrix.transpose(weights_ho)
        #hiddenErrors = who_T * outputErrors

        ## Hidden layer gradiant
        #hiddenGradiants = hiddenOutput * ( 1 - hiddenOutput)
        #hiddenGradiants = hiddenGradiants * hiddenErrors
        #hiddenGradiants = hiddenGradiants * self.lr

        ## Calculate deltas
        #input_t = np.matrix.transpose(input)
        #weights_ih_deltas = hiddenGradiants * input_t

        ## change the weights by the deltas
        #weights_ih += weights_ih_deltas
        ## Change the bias with the gradiant
        #bias_h += hiddenGradiants

        #print (f"{self.weights_output[1]} \n {finalOutputs}")


        #----------------------------------------------------------------------

        # Reverse the lists
        weights = self.weights[::-1] # ho, ih
        weightsO = self.weights_output[::-1] # output, hidden, input 



        firstItter = True
        index = 0
        for w, b, in zip (self.weights[::-1], self.biases[::-1]):

            preWeight = weights[index - 1].T # Get the previous weights to change the next weights
            wo = weightsO[index]    

            if (firstItter == True):
                Error = targetArray.T - wo
            else:
                Error = preWeight * Error

            gradiants = wo # self.activation_function(wo)
            gradiants = gradiants * Error
            gradiants = gradiants * self.lr

            delta = gradiants * weightsO[index+1].T

            # This is the thing that needs to change
            w += delta
            b += gradiants

            firstItter = False

            index += 1

            


            # Error = weight_Output[1] Error

            #gradiant = weights[1]

            #gradiant *= Error
            #gradiant *= lr

