import neuralNetwork as nn
import numpy as np
import random


#with np.load("mnist.npz") as data:
#     training_images = data['training_images']
#     training_labels = data['training_labels']




layer_sizes = (2,2,1)

#x = [np.ones((layer_sizes[0], 1))]

x = [np.asarray([[1],[0]]),     np.asarray([[0],[1]]),    np.asarray([[0],[0]]),  np.asarray([[1],[1]])]
y = [np.asarray([1]),           np.asarray([1]),          np.asarray([0]),        np.asarray([0])]

z = np.ones((layer_sizes[0], 1))

brain = nn.NeuralNetwork(layer_sizes)
for w in brain.weights:
    print (f"{w}\n")

print (brain.predict(x[0]))
print (brain.predict(x[1]))
print (brain.predict(x[2]))
print (brain.predict(x[3]))

for i in range(10000):
    index = random.randint(0,3)
    brain.train(x[index], y[index])


for w in brain.weights:
    print (f"{w}\n")

print (brain.predict(x[0]))
print (brain.predict(x[1]))
print (brain.predict(x[2]))
print (brain.predict(x[3]))
print ("Done")


# for i in range (len(training_images)):
#     print (brain.predict(training_images[0], training_labels[0], training = True))

# print (brain.predict(training_images[i]))

# for i in range (len(training_images)):
#     training = brain.backprop(training_images[i], training_labels[i])
    # print (training)
