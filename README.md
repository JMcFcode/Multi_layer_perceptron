# MNIST Dataset


Optimized for use on the MNIST Dataset provided at:
https://www.kaggle.com/oddrationale/mnist-in-csv

## Multi Layer Perceptron
Multi Layer Perceptron implemented from scratch. Record ~ 75%

### Requirements:
The following packages must be available for import:
```commandline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
```

### References:
http://neuralnetworksanddeeplearning.com/chap2.html

https://www.3blue1brown.com/lessons/backpropagation-calculus

### Usage:
```
from functions_mlp import Network

test_class = Network(list_nodes=[16,10], 
                     act_list=['sigmoid', 'softmax'],
                     l_rate=10)

test_class.sgd(num_per_iter=400, num_iter=150, draw_cost=True)

df_test = test_class.test_model(use_train=False)
df_train = test_class.test_model(use_train=True)
```
The above code initializes the network, trains it on the MNIST dataset and then tests the model on both the training and test datasets.

### Parameters:
Within functions_mlp.py there are several parameters that can be changed.

```
self.learning_rate = 10
self.list_nodes = [20, 10]
self.activation_list = ['sigmoid', 'softmax']
```

self.list_nodes is the number of nodes that will be in the network per layer. To add more layers simply make the list longer.
self.activation_list defines the activation functions for each layer. 
Currently the working choices are between sigmoid, softmax and tanh.

## Convolutional Neural Network

A Convolutional Neural Network is also trained to see how close to 100% accuracy we can get
on the test data. Currently this Network is significantly better than the usual MLP. Record 98.62%.
Currently the parameters are set inside the MNIST class.

### Requirements:

```commandline
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Usage

```commandline
from functions_cnn import MNIST

test_class = MNIST()
test_class.train_net()
test_class.test_network()
```

### References:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://cs231n.github.io/convolutional-networks/
https://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
