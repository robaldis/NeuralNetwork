import neuralNetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']


layer_sizes = (784,16,10)

brain = nn.NeuralNetwork(layer_sizes)

# print (d[0])


for i in range (len(training_images)):
    print (brain.backprop(training_images[i], training_labels[i]))

# print (brain.predict(training_images[i]))

# for i in range (len(training_images)):
#     training = brain.backprop(training_images[i], training_labels[i])
    # print (training)
