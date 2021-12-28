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
        self.learning_rate = 5
        # self.list_nodes = [16, 10]

        self.read_data()
        self.create_network()

    def read_data(self):
        """
        Read in the data from the csv files that should be saved in the same directory as this file.
        """
        self.test_df = pd.read_csv('mnist_test.csv')
        self.train_df = pd.read_csv('mnist_train.csv')

        self.test_matrix, self.test_labels = self.create_matrix(df=self.test_df)
        self.train_matrix, self.train_labels = self.create_matrix(df=self.train_df)

    def create_network(self):
        """
        Manually write in the size of the weights and biases for each layer of the network.
        :return:
        """
        self.w = {'w1': np.random.randn(16, 784),
                  'w2': np.random.randn(10, 16)}

        self.b = {'b1': np.random.randn(16),
                  'b2': np.random.randn(10)}

    # def create_network(self):
    #     """
    #     Automatically create the weights and biases from self.list_nodes.
    #     """
    #     self.w = {}
    #     self.b = {}
    #     for i in range(len(self.list_nodes)):
    #         self.w['w'+str(i)] = np.random.randn(self.list_nodes[i], 784)
    #         self.b['b'+str(i)] = np.random.randn(self.list_nodes[i])

    def feedforward(self, image_data: np.ndarray) -> dict:
        res = {}
        res['a0'] = image_data
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
        :return: two arrays, one with the matrix and one with the label vector!
        """
        matrix = np.array(df.drop(columns=['label'])).T
        labels = np.array(df['label'])
        return matrix, labels

    @staticmethod
    def sigmoid(x_array):
        return 1 / (1 + np.exp(-x_array))

    def sig_deriv(self, x_array: np.ndarray) -> np.ndarray:
        out = self.sigmoid(x_array) * (1 - self.sigmoid(x_array))
        return out

    @staticmethod
    def one_hot(label_vec: np.ndarray) -> np.ndarray:
        """
        One hot encode the labels to make use of them easier later.
        :param label_vec:
        :return: digit_mat
        """
        digit_mat = np.zeros((10, label_vec.size))
        for i in range(len(label_vec)):
            digit_mat[label_vec[i], i] = 1
        digit_mat = digit_mat
        return digit_mat

    def back_prop(self, array_data: np.ndarray, array_labels: np.ndarray) -> dict:
        """
        Calculate the cost vector.
        Backpropogate the errors through the network to adjust both the weights and the biases.
        """
        dict_vals = {'dw2': [], 'db2': [], 'dw1': [], 'db1': [], 'Cost': []}
        for i in range(array_data.shape[1]):
            labels = array_labels[:, i]
            res = self.feedforward(image_data=array_data[:, i])

            delta = self.sig_deriv(res['z2']) * 2 * (res['a2'] - labels)
            db2_array = delta
            dw2_array = np.outer(delta, res['a1'].T)

            delta = np.dot(self.w['w2'].T, delta) * self.sig_deriv(res['z1'])

            db1_array = delta
            dw1_array = np.outer(delta, res['a0'].T)

            dict_vals['dw2'].append(dw2_array)
            dict_vals['db2'].append(db2_array)

            dict_vals['dw1'].append(dw1_array)
            dict_vals['db1'].append(db1_array)

            dict_vals['Cost'].append(np.sum((res['a2'] - labels) ** 2))

        end_vals = {'dw2': self.av_array(dict_vals['dw2']),
                    'db2': self.av_array(dict_vals['db2']),
                    'dw1': self.av_array(dict_vals['dw1']),
                    'db1': self.av_array(dict_vals['db1']),
                    'Average_Cost': np.mean(dict_vals["Cost"])}

        print(f'Average Cost: {end_vals["Average_Cost"]}')

        return end_vals

    @staticmethod
    def av_array(list_array: list) -> np.ndarray:
        """
        Calculate the average of a list of numpy vectors,
        :param list_array:
        :return: np.ndarray:
        """
        out = np.zeros(shape=list_array[0].shape)
        for arr in list_array:
            out += arr
        return 1 / len(list_array) * out

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
            # print(f'Generation {i} complete.')

        if draw_cost:
            plt.plot(range(num_iter), cost_list)
            plt.xlabel('Generation')
            plt.ylabel('Cost')
            plt.title('Average Cost as a function of Generation Trained')

    def test_model(self) -> pd.DataFrame:
        """
        Test out the models accuracy on the test data provided.
        :return: df
        """
        list_corr = []
        list_preds = []
        for i in range(self.test_matrix.shape[1]):
            image = self.test_matrix[:, i]
            actual_val = self.test_labels[i]
            res = self.feedforward(image_data=image)
            pred_val = np.argmax(res['a2'])
            if pred_val == actual_val:
                list_corr.append(1)
            else:
                list_corr.append(0)
            list_preds.append([pred_val, actual_val])
        df = pd.DataFrame(list_preds, columns=['Predictions', 'Actual'])
        print(f'Percentage Correct is: {100 * sum(list_corr) / len(list_corr)}%')
        return df

    def draw_number(self, val: int):
        """
        Method that will draw the number trying to be read.
        Method will also tell you our prediction and the actual value
        :param val:
        """
        image = self.test_matrix[:, val]
        actual_val = self.test_labels[val]
        res = self.feedforward(image_data=image)
        pred_val = np.argmax(res['a2'])
        print(f'Actual Value: {actual_val}')
        print(f'Predicted Value: {pred_val}')
        image = self.test_matrix[:, val].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.show()
