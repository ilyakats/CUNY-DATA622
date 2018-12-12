# -*- coding: utf-8 -*-
"""
CUNY MSDS Program, DATA 622, Final Project
Created: December 2018

@author: Ilya Kats
"""

# Import data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# Import network library
import network622 as nt

# Train basic network to test out setup
# Using cross entropy cost function
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 10, 10, 0.5, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9119, 9232, 9280, 9388, 9394, 9422, 9447, 9466, 9455, 9473
# Best: 94.73%

# Adjust weight initialization
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer() # Not necessary since this is default method
net.SGD(training_data, 10, 10, 0.5, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9349, 9503, 9561, 9561, 9572, 9541, 9558, 9592, 9584, 9583
# Best: 95.92%
# Provides much better starting point

# Adjust cost function from cross entropy to quadratic
net = nt.Network([784, 30, 10], cost=nt.QuadraticCost)
net.default_weight_initializer()
net.SGD(training_data, 10, 10, 0.5, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9253, 9356, 9425, 9443, 9465, 9489, 9504, 9530, 9536, 9551
# Best: 95.51%
# Cross entropy is slightly better

# Add regularization
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data, 10, 10, 0.5, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9384, 9458, 9496, 9511, 9511, 9522, 9555, 9544, 9535, 9584
# Best: 95.84%
# Comparable to model with similar hyper-parameters without regularization

# Increase number of epochs to 30
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 30 epochs:
# 9352, 9371, 9512, 9527, 9324, 9582, 9524, 9606, 9560, 9584,
# 9597, 9504, 9598, 9608, 9638, 9553, 9580, 9646, 9574, 9580,
# 9588, 9577, 9596, 9645, 9572, 9627, 9635, 9633, 9579, 9606
# Best: 96.46%
# Best result so far, but training stagnated pretty quickly

# Decrease number of epochs back to 10 to speed up training,
# but increase hidden layer to 100 neurons
net = nt.Network([784, 100, 10], cost=nt.CrossEntropyCost)
net.default_weight_initializer()
net.SGD(training_data, epochs=10, mini_batch_size=10, 
        eta=0.5, lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs
# 9521, 9650, 9689, 9701, 9684, 9727, 9720, 9665, 9706, 9759
# Best: 97.59%
# Significant improvement

# AT THIS POINT NETWORK CLASS WAS MODIFIED TO INCLUDE ReLU
# After ReLU was added, hyper-parameters had to be adjusted to produce 
# decent results. This is especially true for learning rate which 
# through trial-and-error was adjusted from 0.5 to 0.05.
# Number of hidden neuron also plays important role in accuracy. 

# Typical run of 10 epochs with 39 hidden neurons
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9287, 9370, 9445, 9482, 9508, 9452, 9493, 9512, 9511, 9517
# Best: 95.17%

# Testing if ReLU can beat Sigmoid with 100 hidden neurons
net = nt.Network([784, 100, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9603, 9679, 9743, 9716, 9739, 9746, 9782, 9748, 9754, 9764
# Best: 97.82% 
# Only slightly better than sigmoid network

# Forgot to test regularization with ReLU
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        lmbda = 5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9145, 9460, 9465, 9510, 9536, 9540, 9576, 9582, 9553, 9571
# Best: 95.82%
# Rocky start, but noticeably better than without regularization

# Different lambda
net = nt.Network([784, 30, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 10, 5, 0.05, 
        lmbda = 50,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)
# Accuracy data for 10 epochs:
# 9404, 9480, 9534, 9467, 9518, 9536, 9548, 9541, 9549, 9577
# Best: 95.77%
# Different learning progress, but similar outcome

# THE FOLLOWING INCLUDES CODE FOR INVESTIGATING MODEL OUTPUT, 
# INCLUDING SOFTMAX FUNCTION

# Code for displaying a single MNIST digit
img_index = 54
from matplotlib import pyplot as plt
import numpy as np
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt
gen_image(test_data[img_index][0]).show()

# Testing out calculating and accessing predictions and actual values
# Prediction
net.feedforward(test_data[img_index][0])
net.predict(test_data[img_index][0])
# Actual
np.argmax(training_data[img_index][1])
test_data[img_index][1]

# Loop through test data and gather all mis-classifications
err = []
for pixels, digit in test_data:
    p = net.feedforward(pixels)
    if (digit != np.argmax(p)):
        err.append([pixels,digit,p])

# Sample wrong prediction
gen_image(err[1][0]).show() # Image
np.argmax(err[1][2])        # Predicted digit
np.round(err[1][2], 4)      # NN output array
err[1][1]                   # Actual digit

# Softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Softmax input
softmax(net.feedforward(test_data[img_index][0]))
softmax(err[1][2])
softmax(err[1][2]*10)

# ADDING VARIABLE LEARNING RATE TO THE NETWORK CODE

import datetime
print(datetime.datetime.now())
net = nt.Network([784, 100, 10], cost=nt.CrossEntropyCost, neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 60, 5, 0.05, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        early_stopping_n=4,
        variable_learning_coef=0.5)
print(datetime.datetime.now())

"""
FINAL RUN:

2018-12-10 00:52:55.580108
Epoch 0 training complete
Accuracy on evaluation data: 9546 / 10000
Epoch 1 training complete
Accuracy on evaluation data: 9617 / 10000
Epoch 2 training complete
Accuracy on evaluation data: 9705 / 10000
Epoch 3 training complete
Accuracy on evaluation data: 9689 / 10000
Epoch 4 training complete
Accuracy on evaluation data: 9723 / 10000
Epoch 5 training complete
Accuracy on evaluation data: 9750 / 10000
Epoch 6 training complete
Accuracy on evaluation data: 9733 / 10000
Epoch 7 training complete
Accuracy on evaluation data: 9752 / 10000
Epoch 8 training complete
Accuracy on evaluation data: 9753 / 10000
Epoch 9 training complete
Accuracy on evaluation data: 9747 / 10000
Epoch 10 training complete
Accuracy on evaluation data: 9763 / 10000
Epoch 11 training complete
Accuracy on evaluation data: 9775 / 10000
Epoch 12 training complete
Accuracy on evaluation data: 9773 / 10000
Epoch 13 training complete
Accuracy on evaluation data: 9765 / 10000
Epoch 14 training complete
Accuracy on evaluation data: 9774 / 10000
Epoch 15 training complete
Accuracy on evaluation data: 9787 / 10000
Epoch 16 training complete
Accuracy on evaluation data: 9788 / 10000
Epoch 17 training complete
Accuracy on evaluation data: 9778 / 10000
Epoch 18 training complete
Accuracy on evaluation data: 9773 / 10000
Epoch 19 training complete
Accuracy on evaluation data: 9779 / 10000
Epoch 20 training complete
Accuracy on evaluation data: 9779 / 10000
New learning rate: 0.0125
Epoch 21 training complete
Accuracy on evaluation data: 9793 / 10000
Epoch 22 training complete
Accuracy on evaluation data: 9792 / 10000
Epoch 23 training complete
Accuracy on evaluation data: 9788 / 10000
Epoch 24 training complete
Accuracy on evaluation data: 9794 / 10000
Epoch 25 training complete
Accuracy on evaluation data: 9793 / 10000
Epoch 26 training complete
Accuracy on evaluation data: 9790 / 10000
Epoch 27 training complete
Accuracy on evaluation data: 9791 / 10000
Epoch 28 training complete
Accuracy on evaluation data: 9791 / 10000
New learning rate: 0.00625
Epoch 29 training complete
Accuracy on evaluation data: 9794 / 10000
Epoch 30 training complete
Accuracy on evaluation data: 9797 / 10000
Epoch 31 training complete
Accuracy on evaluation data: 9794 / 10000
Epoch 32 training complete
Accuracy on evaluation data: 9792 / 10000
Epoch 33 training complete
Accuracy on evaluation data: 9786 / 10000
Epoch 34 training complete
Accuracy on evaluation data: 9793 / 10000
New learning rate: 0.003125
Epoch 35 training complete
Accuracy on evaluation data: 9794 / 10000
Epoch 36 training complete
Accuracy on evaluation data: 9798 / 10000
Epoch 37 training complete
Accuracy on evaluation data: 9797 / 10000
Epoch 38 training complete
Accuracy on evaluation data: 9790 / 10000
Epoch 39 training complete
Accuracy on evaluation data: 9793 / 10000
Epoch 40 training complete
Accuracy on evaluation data: 9801 / 10000
Epoch 41 training complete
Accuracy on evaluation data: 9798 / 10000
Epoch 42 training complete
Accuracy on evaluation data: 9793 / 10000
Epoch 43 training complete
Accuracy on evaluation data: 9794 / 10000
Epoch 44 training complete
Accuracy on evaluation data: 9795 / 10000
New learning rate: 0.0015625
Epoch 45 training complete
Accuracy on evaluation data: 9798 / 10000
Epoch 46 training complete
Accuracy on evaluation data: 9798 / 10000
Epoch 47 training complete
Accuracy on evaluation data: 9794 / 10000
Epoch 48 training complete
Accuracy on evaluation data: 9790 / 10000
2018-12-10 01:24:07.102238
"""

# Just for quick reference evaluate network with 2 hidden layers
# (30 and 30 neurons) and same hyper-parameters. 
print(datetime.datetime.now())
net = nt.Network([784, 30, 30, 10], 
                 cost=nt.CrossEntropyCost, 
                 neuron=nt.ReLUNeuron)
net.default_weight_initializer()
net.SGD(training_data, 60, 5, 0.05, 
        lmbda=5,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        early_stopping_n=4,
        variable_learning_coef=0.5)
print(datetime.datetime.now())
# Stopped training after 32 epochs. 
# Best: 96.57%
