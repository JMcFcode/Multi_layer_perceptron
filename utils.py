"""
Python file to host shared functions.
"""
import pandas as pd
import numpy as np


def create_matrix(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    This is for turning the original dataset into useful values.
    :param df:
    :return: two arrays, one with the matrix and one with the label vector!
    """
    matrix = np.array(df.drop(columns=['label'])).T
    labels = np.array(df['label'])
    return matrix, labels


def matrix_3d(array: np.ndarray) -> np.ndarray:
    """
    Turn the orginal vector into an array of 3D images
    :param array:
    :return: out_array
    """
    out_array = []
    for i in range(array.shape[1]):
        col = array[:, i]
        img = col.reshape((28, 28))
        out_array.append(img)
    return np.array(out_array)


def read_data(csv_path: str) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(csv_path)
    matrix, labels = create_matrix(df=df)
    matrix = matrix / np.amax(matrix)
    return matrix, labels


def flat_list(t: list) -> list:
    return [item for sublist in t for item in sublist]