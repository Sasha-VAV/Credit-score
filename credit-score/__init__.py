from sklearn.pipeline import Pipeline
from .preprocessors import BasePreprocessor
from .models import LogisticRegression

"""
pipeline = Pipeline(steps=[
    ('preprocessor', BasePreprocessor()),
    ('model', LogisticClassifier())
])
"""
