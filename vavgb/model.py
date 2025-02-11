import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def cross_entropy(y_true, y_pred):
    



class GradientBoosting(BaseEstimator):
    def __init__(self, criterion: str = 'mse', n_estimators: int = 10, max_depth: int = 2, lr: float = 0.1):
        """
        Initializes a GradientBoosting estimator.
        :param criterion: function, either cross-entropy or mse
        :param n_estimators: number of estimators
        :param max_depth: maximum depth of decision tree
        :param lr: learning rate
        """
        criterions = {
            'mse': mean_squared_error,
        }
        self.criterion = criterions[criterion]
        self.n_estimators = n_estimators
        self.lr = lr
        self.estimators = list()


    def fit(self, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        if self.criterion == 'mse':
            self.estimators.append(y.mean())
        else:
            a = y.min()
            b = y.max()
            c = (a+b)/2
            while self.criterion(c) > 1e-3:
                ...
        for i in range(1, self.n_estimators):
            ...




    def predict(self, X):
        pass

def f(x):
    return math.sin(x)

X = np.array([i for i in range(0, 100)]) / 20
Y = np.array([f(X[i]) for i in range(0, 100)])
Y_pred = np.random.rand(100)
#plt.plot(X, Y)

#plt.show()

Y = torch.from_numpy(Y).float().reshape(1,Y.shape[0], 1)
Y_pred = torch.from_numpy(Y_pred).float().reshape(1,Y_pred.shape[0], 1)
loss = nn.CrossEntropyLoss()
print(loss(Y_pred, Y))
