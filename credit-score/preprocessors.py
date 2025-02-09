from sklearn.base import TransformerMixin
import pandas as pd


class BasePreprocessor(TransformerMixin):
    def __init__(self):
        self.means = None
        self.stds = None
        self.medians = None

    def fit(self, X: pd.DataFrame) -> 'BasePreprocessor':
        self.means = X.mean()
        self.stds = X.std()
        self.medians = X.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        answer = X.fillna(self.medians)
        answer = (answer - self.means) / self.stds
        return answer

