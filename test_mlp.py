#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 08:50:02 2021

@author: joelmcfarlane
"""

from functions_mlp import Network

test_class = Network()

train_df = test_class.train_df
test_df = test_class.test_df

matrix_train = test_class.train_matrix
label_train = test_class.train_labels


a3 = test_class.feedforward(matrix_train[:,0])

cost = test_class.cost_function(actual_vec=a3, digit=label_train[0])


#%%
import matplotlib.pyplot as plt
# pick a sample to plot
sample = 4
image = matrix_train[:,sample].reshape(28,28)# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()

#%% - Test iterations:
from functions_mlp import Network

test_class = Network()

res = test_class.back_propogation(iter=0)
#%% - test traning

from functions_mlp import Network

test_class = Network(list_nodes=[16,10], 
                     act_list=['sigmoid', 'softmax'],
                     l_rate=10)

test_class.sgd(num_per_iter=400, num_iter=150, draw_cost=True)

df_test = test_class.test_model(use_train=False)
df_train = test_class.test_model(use_train=True)


#%% - Look at specific examples:
import random
# val = random.randint(0,1000)

val = 102
test_class.draw_number(val=val)
