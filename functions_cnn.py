"""
Created on Wed Jan 5 2022

@author: joelmcfarlane
"""
import pandas as pd
import torch.nn as nn
import torch.nn.functional as func
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils

sns.set()


class MyDataset(Dataset):
    """
    Has Three Required methods, all represented here.
    """
    def __init__(self, data: np.ndarray, target: np.ndarray, transform=None):
        data = utils.matrix_3d(data)
        data = np.expand_dims(data, 1)
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        self.len = len(data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return self.len


class CNN(nn.Module):
    """
    Create a Pytorch CNN
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5), stride=1, padding=0)
        self.mlp1 = nn.Linear(20 * 4 * 4, 50)
        self.mlp2 = nn.Linear(50, 60)
        self.mlp3 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = func.relu(self.mlp1(x))
        x = func.relu(self.mlp2(x))
        x = func.softmax(self.mlp3(x))
        return x


class MNIST:
    """
    Test and Train the MNIST Dataset
    """

    def __init__(self):
        self.batch_size = 100
        self.num_workers = 5
        self.net = CNN()
        self.epochs = 5
        self.learning_rate = 0.05
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)

        self.get_data()

    def get_data(self):
        test_matrix, test_labels = utils.read_data('mnist_test.csv')
        testing_data = MyDataset(data=test_matrix, target=test_labels)
        self.test_dataloader = DataLoader(testing_data, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers)

        train_matrix, train_labels = utils.read_data('mnist_train.csv')
        training_data = MyDataset(data=train_matrix, target=train_labels)
        self.train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def train_net(self, plot_graph=False):

        list_cost = []

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                list_cost.append(loss.item())
                if i % 10 == 0:
                    print(f'Loss: {round(loss.item(), 4)}, Epoch: {epoch+1}')

        print('Finished Training')

        if plot_graph:
            x_scale = np.arange(len(list_cost))
            plt.plot(x_scale, list_cost,
                     label='\u03B1 :' + str(self.learning_rate) + f'\n Epochs: {self.epochs}')
            plt.xlabel('Generation')
            plt.ylabel('Cost')
            plt.legend(loc='best')

    def test_network(self) -> pd.DataFrame:
        correct = 0
        total = 0

        list_pred = []
        list_act = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                list_act.append([tens.item() for tens in labels])
                list_pred.append([tens.item() for tens in predicted])

        df = pd.DataFrame(data={'Predicted': utils.flat_list(list_pred),
                                'Actual': utils.flat_list(list_act)})
        print(f'Test Data Accuracy: {round(100 * correct / total, 2)}%')
        df['Correct'] = df['Predicted'] == df['Actual']
        return df

    def plot_example(self):
        pass
