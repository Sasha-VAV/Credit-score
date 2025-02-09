from .preprocessors import BasePreprocessor
from .models import LogisticRegression, KNNClassifier
from .metrics import f1_score, cross_val_score
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
import pandas as pd


def main():
    data = pd.read_csv('data/cs-training.csv')
    target_column = 'SeriousDlqin2yrs'
    Y = data[target_column]
    X = data.drop(target_column, axis=1)
    preprocessor = BasePreprocessor()
    X = preprocessor.fit_transform(X)
    scores = cross_val_score(KNNClassifier(), X, Y)
    print(scores)


if __name__ == '__main__':
    main()
