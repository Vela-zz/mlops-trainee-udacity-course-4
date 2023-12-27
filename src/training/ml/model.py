import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score

import aequitas
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.plotting import Plot
from aequitas.fairness import Fairness
from aequitas.preprocessing import preprocess_input_df


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = DecisionTreeClassifier()
    cv = KFold(5)
    trainer = GridSearchCV(clf,
                           {"max_depth": np.linspace(5, 30, 6).astype(int)},
                           cv=cv)
    trainer.fit(X_train, y_train)
    bst_model = trainer.best_estimator_
    return bst_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {"precision": precision,
            "recall": recall,
            "fbeta": fbeta}


def plot_model_disparity_on_fpr(data: pd.DataFrame, output_path: str):
    """calculate model's disparity on each catgorical features.

    Args:
        data (pd.DataFrame): a pandas dataframe with catgorical features and model's predict result  # NOQA:E501
        output (str): a path of folder to save disparity on graph
    """
    df, _ = preprocess_input_df(data)
    g = Group()
    aqp = Plot()
    xtab, _ = g.get_crosstabs(df)

    figure, ax = plt.subplots(1, 1, figsize=(12, 32))
    _ = aqp.plot_group_metric(xtab, 'fpr', ax=ax)
    figure.savefig(os.path.join(output_path, "fpr_fiarness_graph.png"))


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
