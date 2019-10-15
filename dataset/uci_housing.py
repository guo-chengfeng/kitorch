"""
UCI Housing dataset.

This module will download dataset from
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ and
parse training set and test set into paddle reader creators.
"""
from __future__ import print_function

import numpy as np
import six
from . import common

URL = 'http://paddlemodels.bj.bcebos.com/uci_housing/housing.data'
MD5 = 'd4accdce7a25600298819f8e28e8d593'
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'convert'
]

UCI_TRAIN_DATA = None
UCI_TEST_DATA = None
FILE_NAME = None


def feature_range(maximums, minimums):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    feature_num = len(maximums)
    ax.bar(list(range(feature_num)),
           maximums - minimums,
           color='r',
           align='center')
    ax.set_title('feature scale')
    plt.xticks(list(range(feature_num)), feature_names)
    plt.xlim([-1, feature_num])
    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.show()


def fetch(feature_num=14, ratio=0.8, shuffle=False, normalized=True, visualized=True):
    global UCI_TRAIN_DATA, UCI_TEST_DATA,FILE_NAME

    FILE_NAME = common.download(URL, 'uci_housing', MD5)
    data = np.fromfile(FILE_NAME, sep=' ')
    data = data.reshape(data.shape[0] // feature_num, feature_num)

    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    if visualized:
        feature_range(maximums[:-1], minimums[:-1])
    if normalized:
        for i in six.moves.range(feature_num - 1):
            data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    offset = int(data.shape[0] * ratio)
    UCI_TRAIN_DATA = data[:offset]
    UCI_TEST_DATA = data[offset:]

    if shuffle:
        np.random.shuffle(UCI_TRAIN_DATA)
        np.random.shuffle(UCI_TEST_DATA)

    return UCI_TRAIN_DATA, UCI_TEST_DATA

