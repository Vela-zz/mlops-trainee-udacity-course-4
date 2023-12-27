import os
import logging
import argparse
from pathlib import Path

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (train_model, compute_model_metrics)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_data(file_path: str):
    """this function load data and split it into train, test

    Args:
        file_path (str): the path of data
    """
    data = pd.read_csv(file_path)
    return data


def training(input_path: str, output_path: str):
    runtime_path = Path(os.getcwd()).parent.parent
    input_path = os.path.join(runtime_path, input_path)
    output_path = os.path.join(runtime_path, output_path)

    data = load_data(input_path)
    logger.info("STEP[train]: (1/3) Data Loaded.")
    # generate k-fold
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # train, test split
    train_data, test_data = train_test_split(data, test_size=0.2)

    X_train, y_train, encoder, lb = process_data(train_data, categorical_features=cat_features,  # NOQA:E501
                                                 label="salary", training=True)

    model = train_model(X_train, y_train)
    logger.info("STEP[train]: (2/3) Train Completed.")
    # eval on test dataset
    X_test, y_test, _, _ = process_data(test_data, categorical_features=cat_features,  # NOQA:E501
                                        label='salary', training=False,
                                        encoder=encoder, lb=lb)
    y_pred = model.predict(X_test)
    scores = compute_model_metrics(y_test, y_pred)
    logger.info("STEP[train]: (3/3) Eval Completed.")
    # save model and score
    with open(os.path.join(output_path, 'dct_model.pkl'), 'wb') as mf:
        pickle.dump(model, mf)

    with open(os.path.join(output_path, 'model_score.txt'), "w") as f:
        for eval_metric, metric_score in scores.items():
            f.write(f"{eval_metric}: {metric_score:.3f}\n")
    logger.info("STEP[train]: Final Result Saved.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument("--input_path",
                        type=str,
                        help="path of the raw data stored",
                        required=True)

    parser.add_argument("--output_path",
                        type=str,
                        help="path of the model need to be output",
                        required=True)

    args = parser.parse_args()

    training(args.input_path, args.output_path)