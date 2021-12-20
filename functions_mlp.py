#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 08:46:58 2021

@author: joelmcfarlane

Building a multi_layer_perceptron from the ground up
"""

import numpy as np
import pandas as pd


class Network:
    """
    We will use a neural network to try and create a model that can read 
    numbers.
    We will use the sigmoid function as our activation function.
    """

    def __init__(self):
        self.alpha = 1
        self.dict_network = {}
        self.read_data()
        self.create_network()
        self.iter_size = 50

    def read_data(self):
        self.test_df = pd.read_csv('mnist_test.csv')
        self.train_df = pd.read_csv('mnist_train.csv')

        self.test_matrix, self.test_labels = self.create_matrix(df=self.test_df)
        self.train_matrix, self.train_labels = self.create_matrix(df=self.train_df)

    def create_network(self):
        self.w1 = np.random.rand(16, 784) - 0.5
        self.w2 = np.random.rand(16, 16) - 0.5
        self.w3 = np.random.rand(10, 16) - 0.5

        self.b1 = np.random.rand(16) - 0.5
        self.b2 = np.random.rand(16) - 0.5
        self.b3 = np.random.rand(10) - 0.5

    def feedforward(self, image_data: np.ndarray):
        z1 = np.dot(self.w1, image_data) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(self.w3, a2) + self.b3
        a3 = self.sigmoid(z3)
        return a3, a2, a1, z1, z2, z3

    @staticmethod
    def create_matrix(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        matrix = np.array(df.drop(columns=['label'])).T
        labels = np.array(df['label'])
        return matrix, labels

    @staticmethod
    def sigmoid(x_array):
        return 1 / (1 + np.exp(-x_array))

    @staticmethod
    def sig_deriv(x_array: np.ndarray) -> np.ndarray:
        return np.exp(-x_array) / (1 + np.exp(-x_array)) ** 2

    @staticmethod
    def one_hot(label_vec: np.ndarray) -> np.ndarray:
        digit_mat = np.zeros((10, label_vec.size))
        for i in range(len(label_vec)):
            digit_mat[label_vec-1,i] = 1
        digit_mat = digit_mat.T
        return digit_mat

    def back_propogation(self, iter: float):
        sample_train = self.train_matrix[:, iter:iter + 50]
        sample_labels = self.train_labels[iter:iter + 50]
        num_per_iter = self.iter_size
        a0_list = np.zeros(num_per_iter)
        a1_list = np.zeros(num_per_iter)
        a2_list = np.zeros(num_per_iter)
        a3_list = np.zeros(num_per_iter)
        z1_list = np.zeros(num_per_iter)
        z2_list = np.zeros(num_per_iter)
        z3_list = np.zeros(num_per_iter)
        one_hot_y = self.one_hot(sample_labels)
        for i in range(num_per_iter):
            a0 = sample_train[:, i]
            actual_vec, a2, a1, z1, z2, z3 = self.feedforward(image_data=a0)
            a0_list[i] = a0
            a1_list[i] = a1
            a2_list[i] = a2
            a3_list[i] = actual_vec
            z1_list[i] = z1
            z2_list[i] = z2
            z3_list[i] = z3

        dcdw3 = 1 / num_per_iter * np.sum(2 * np.dot(a2_list, self.sig_deriv(z3_list)).dot(a3_list - one_hot_y))
        dcdb3 = 1 / num_per_iter * np.sum(2 * np.dot(self.sig_deriv(z3_list), a3_list - one_hot_y))

        dcdw2 = 1 / num_per_iter * np.sum(2 * np.dot(a1_list, self.sig_deriv(z2_list)).dot(a2_list - ))
        dcdb2 = 1 / num_per_iter * np.sum(2 * np.dot(self.sig_deriv(z2_list), a2_list - one_hot_y))

        dcdw1 = 1 / num_per_iter * np.sum(2 * np.dot(a0_list, self.sig_deriv(z1_list)).dot(a1_list - one_hot_y))
        dcdb1 = 1 / num_per_iter * np.sum(2 * np.dot(self.sig_deriv(z1_list), a1_list - one_hot_y))

        return dcdw3, dcdb3, dcdw2, dcdb2, dcdw1, dcdb1

    def back_prop(self):
        """
        Matrix version with no for loop.
        :return:
        """


    def update_params(self, dw3: float, db3: float, dw2: float, db2: float, dw1: float, db1: float):
        self.w1 = self.w1 - self.alpha * dw1
        self.w2 = self.w2 - self.alpha * dw2
        self.w3 = self.w3 - self.alpha * dw3
        self.b1 = self.b1 - self.alpha * db1
        self.b2 = self.b2 - self.alpha * db2
        self.b3 = self.b3 - self.alpha * db3

    def test_model(self):
        pass

    def draw_number(self):
        pass
