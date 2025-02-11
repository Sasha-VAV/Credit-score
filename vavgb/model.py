import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class GradientBoosting(BaseEstimator):
    def __init__(self, criterion: str = 'mse', n_estimators: int = 10, max_depth: int = 2, lr: float = 0.3):
        """
        Initializes a GradientBoosting estimator.
        :param criterion: function, either cross-entropy or mse
        :param n_estimators: number of estimators
        :param max_depth: maximum depth of decision tree
        :param lr: learning rate
        """
        criterions = {
            'mse': nn.MSELoss(reduction='mean'),
            'entropy': nn.CrossEntropyLoss(reduction='mean'),
        }
        self.criterion_name = criterion
        self.criterion = criterions[criterion]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.estimators = list()
        self.n_classes = 1


    def fit(self, X, y):
        n = X.shape[0]
        y = np.array(y).reshape(-1, 1)
        self.n_classes = y.shape[1]
        self.estimators.clear()
        y = torch.from_numpy(y).float().reshape(1, n, self.n_classes)
        if self.criterion_name == 'mse':
            self.estimators.append(y.mean())
        else:
            temp = y.mean()

            self.estimators.append(temp.detach().numpy())

        for _ in tqdm(range(self.n_estimators)):
            y_pred = self.predict(X)
            dt = DecisionTreeRegressor(max_depth=self.max_depth)
            y_pred = torch.from_numpy(y_pred).float().reshape(1, n, self.n_classes)
            y_pred.requires_grad = True
            if self.criterion_name == 'mse':
                loss = self.criterion(y, y_pred)
                y_new = -torch.autograd.grad(loss, y_pred)[0]
                y_new = np.array(y_new.reshape(-1, self.n_classes))
            else:
                y_new = (y-y_pred).reshape(-1, self.n_classes).detach().numpy()
            dt.fit(X, y_new)
            self.estimators.append(dt)

    def predict(self, X):
        y_pred = np.full((X.shape[0], 1), self.estimators[0])
        for estimator in self.estimators[1:]:
            y_pred += estimator.predict(X).reshape(-1, 1) * self.lr
        if self.criterion_name == 'mse':
            return y_pred
        return torch.sigmoid(torch.tensor(y_pred)).detach().numpy()

    def predict_proba(self, X):
        y_pred = self.predict(X)
        return np.concatenate((1-y_pred, y_pred), axis=1)


def main():
    def f(x):
        return math.sin(x)

    X = np.array([i for i in range(0, 100)]) / 20
    Y = np.array([f(X[i]) + (random.random() - 0.5) / 4 for i in range(0, 100)])
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    gb = GradientBoosting(max_depth=4, n_estimators=400)
    gb.fit(X, Y)
    Y_pred = gb.predict(X)
    plt.plot(X, Y)
    plt.plot(X, Y_pred)
    plt.savefig('plot.png')


if __name__ == '__main__':
    main()