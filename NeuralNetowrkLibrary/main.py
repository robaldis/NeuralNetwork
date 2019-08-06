import neuralNetwork as nn
from numpy import *
import random
from time import time, localtime, ctime


def main():
    brain = nn.NeuralNetwrok(2, 2, 1, 0.1)


    input =     [1,0],  [0,1],  [0,0],  [1,1]
    target =    [1],    [1],    [0],    [0]

    startTime = time()

    print("Starting to train the network")
    print(f"Time Started: {ctime(startTime)}")

    for i in range (100000):
        index = random.randint(0,3)
        brain.gradientDecent(input[index], target[index])

    print("Done!!")
    endTime = time()
    timeTaken = endTime - startTime

    print(f"Time ended: {ctime(endTime)}")
    print(f"Time Taken: {round((timeTaken / 60), 2)} minutes\n")

    output = brain.feedForward([0,0])
    if (output > 0.5):
        print("TRUE")
    else:
        print ("FALSE")

    output1 = brain.feedForward([0,1])
    output2 = brain.feedForward([1,0])
    output3 = brain.feedForward([0,0])
    output4 = brain.feedForward([1,1])
    print(f'{output1}\n{output2}\n{output3}\n{output4}')


if (__name__ == "__main__"):
    main()
