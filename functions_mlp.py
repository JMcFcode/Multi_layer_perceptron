#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 08:46:58 2021

@author: joelmcfarlane

Building a multi_layer_perceptron from the ground up
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


class Network:
    """
    We will use a neural network to try and create a model that can read 
    numbers.
    We will use the sigmoid function as our activation function.
    We will have a 2 layer network.
    """

    def __init__(self):
        self.alpha = 1
        self.dict_network = {}
        self.read_data()
        self.create_network()
        self.iter_size = 50
        self.learning_rate = 0.2

    def read_data(self):
        self.test_df = pd.read_csv('mnist_test.csv')
        self.train_df = pd.read_csv('mnist_train.csv')

        self.test_matrix, self.test_labels = self.create_matrix(df=self.test_df)
        self.train_matrix, self.train_labels = self.create_matrix(df=self.train_df)

    def create_network(self):
        self.w = {'w1': np.random.rand(16, 784) - 0.5,
                  'w2': np.random.rand(10, 16) - 0.5}

        self.b = {'b1': np.random.rand(16) - 0.5,
                  'b2': np.random.rand(10) - 0.5}

    def feedforward(self, image_data: np.ndarray):
        res = {}
        res['image'] = image_data
        res['z1'] = np.dot(self.w['w1'], image_data) + self.b['b1']
        res['a1'] = self.sigmoid(res['z1'])

        res['z2'] = np.dot(self.w['w2'], res['a1']) + self.b['b2']
        res['a2'] = self.sigmoid(res['z2'])

        return res

    @staticmethod
    def create_matrix(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        This is for turning the original dataset into useful values.
        :param df:
        :return: two arrays
        """
        matrix = np.array(df.drop(columns=['label'])).T
        labels = np.array(df['label'])
        return matrix, labels

    @staticmethod
    def sigmoid(x_array):
        return 1 / (1 + np.exp(-x_array))

    @staticmethod
    def sig_deriv(x_array: np.ndarray) -> np.ndarray:
        out = np.exp(x_array) / np.power((1 + np.exp(x_array)), 2)
        return np.nan_to_num(out)

    @staticmethod
    def one_hot(label_vec: np.ndarray) -> np.ndarray:
        digit_mat = np.zeros((10, label_vec.size))
        for i in range(len(label_vec)):
            digit_mat[label_vec[i] - 1, i] = 1
        digit_mat = digit_mat
        return digit_mat

    def back_prop(self, array_data: np.ndarray, array_labels: np.ndarray) -> dict:
        """
        Calculate the cost vector.
        """
        dict_vals = {'dw2': [], 'db2': [], 'dw1': [], 'db1': [], 'Cost': []}
        for i in range(array_data.shape[1]):
            labels = array_labels[:, i]
            res = self.feedforward(image_data=array_data[:, i])

            db2_array = self.sig_deriv(res['z2']) * 2 * (res['a2'] - labels)
            dw2_array = np.outer(db2_array, res['a1'])

            delta_1 = np.dot(self.w['w2'].T, 2 * (res['a2'] - labels) * self.sig_deriv(res['z2']))

            db1_array = self.sig_deriv(res['z1']) * delta_1
            dw1_array = np.outer(db1_array, res['image'])

            dict_vals['dw2'].append(dw2_array)
            dict_vals['db2'].append(db2_array)

            dict_vals['dw1'].append(dw1_array)
            dict_vals['db1'].append(db1_array)

            dict_vals['Cost'].append(np.sum(np.power((res['a2'] - labels), 2)))

        end_vals = {}
        end_vals['dw2'] = 1 / len(array_data) * np.sum(dict_vals['dw2'])
        end_vals['db2'] = 1 / len(array_data) * np.sum(dict_vals['db2'])

        end_vals['dw1'] = 1 / len(array_data) * np.sum(dict_vals['dw1'])
        end_vals['db1'] = 1 / len(array_data) * np.sum(dict_vals['db1'])

        end_vals['Average_Cost'] = np.mean(dict_vals["Cost"])
        print(f'Average Cost: {end_vals["Average_Cost"]}')

        return end_vals

    def sgd(self, num_per_iter: int, num_iter: int, draw_cost=False):
        """
        Implement Stochastic Gradient Descent Algo.
        :param draw_cost: Option to Draw function of cost.
        :param num_per_iter:
        :param num_iter:
        """
        cost_list = []
        for i in range(num_iter):
            array_data = self.train_matrix[:, num_iter: num_iter + num_per_iter]
            array_labels = self.one_hot(self.train_labels[num_iter: num_iter + num_per_iter])

            vals_dict = self.back_prop(array_data=array_data, array_labels=array_labels)

            self.w['w1'] = self.w['w1'] - vals_dict['dw1'] * self.learning_rate
            self.b['b1'] = self.b['b1'] - vals_dict['db1'] * self.learning_rate
            self.w['w2'] = self.w['w2'] - vals_dict['dw2'] * self.learning_rate
            self.b['b2'] = self.b['b2'] - vals_dict['db2'] * self.learning_rate

            cost_list.append(vals_dict['Average_Cost'])
            print(f'Generation {i} complete.')

        if draw_cost:
            plt.plot(range(num_iter), cost_list)
            plt.xlabel('Generation')
            plt.ylabel('Cost')
            plt.title('Average Cost as a function of Generation Trained')

    def test_model(self):
        for image in self.test_matrix:
            pass

    def draw_number(self):
        pass
