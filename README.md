# Multi layer perceptron

Multi Layer Perceptron implemented from scratch.
Optimized for use on the MNIST Dataset provided at:
https://www.kaggle.com/oddrationale/mnist-in-csv

## Usage:
```
from functions_mlp import Network

test_class = Network()

test_class.sgd(num_per_iter=400, num_iter=150, draw_cost=True)

df_test = test_class.test_model(use_train=False)
df_train = test_class.test_model(use_train=True)
```
The above code initializes the network, trains it on the MNIST dataset and then tests the model on both the training and test datasets.

## Parameters:
Within functions_mlp.py there are several parameters that can be changed.

```
self.learning_rate = 10
self.list_nodes = [20, 10]
self.activation_list = ['sigmoid', 'softmax']
```

self.list_nodes is the number of nodes that will be in the network per layer. To add more layers simply make the list longer.
self.activation_list defines the activation functions for each layer. Currently the working choices are between sigmoid, softmax and tanh.
