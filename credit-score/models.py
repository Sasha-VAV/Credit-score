import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from .functions import log_loss
import pandas as pd


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, X, y, lr=0.01, epochs=20, batch_size=100):
        self.W = torch.rand((X.shape[1]))
        self.b = torch.rand(1)
        batches = X.shape[0] // batch_size
        for i in range(epochs):
            running_loss = 0.0
            for j in range(batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                x_batch = X[start:end]
                y_batch = y[start:end]
                y_pred = self.predict(x_batch)
                #loss = torch.mean((y_batch - y_pred) ** 2)
                loss = log_loss(y_pred, y_batch)
                grad = (y_batch - y_pred)
                #grad = -2 * (y_batch - y_pred) * y_pred * (1 - y_pred)
                grad_w = x_batch.T @ grad / batch_size

                self.W -= lr * grad_w
                self.b -= lr * torch.mean(grad)
                running_loss += loss.item()
            print('Epoch: {}, Loss: {}'.format(i, running_loss))

    def predict(self, X):
        return torch.sigmoid(X@self.W + self.b).flatten()


class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.neighbors_ = None
        self.y = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        self.neighbors_ = X
        self.y = y

    def predict(self, X):
        pred = torch.zeros(X.shape[0])
        for i in tqdm(range(X.shape[0])):
            temp = torch.linalg.norm(self.neighbors_ - X[i], axis=1)
            temp = torch.argsort(temp)[:self.n_neighbors]
            temp = self.y[temp]
            pred[i] = torch.mode(temp)[0]




