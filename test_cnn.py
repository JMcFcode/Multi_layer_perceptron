"""
Created on Wed Jan 5 2022

@author: joelmcfarlane
"""

#%% - Test Loading Data
from functions_cnn import MyDataset
from utils import read_data, matrix_3d

matrix, labels = read_data('mnist_train.csv')
data_class = MyDataset(data=matrix, target=labels)

matrix_test = matrix_3d(matrix)
#%% - Initiliase CNN:

from functions_cnn import CNN

cnn_class = CNN()


#%%- Test MNSIT Class
from functions_cnn import MNIST

test_class = MNIST()

test_class.train_net(plot_graph=True)

cnn_res, perc_corr = test_class.test_network()
