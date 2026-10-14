
import argparse
import json
from csv import DictReader
from vectorizer import Vectorizer
from logistic_regression import LogisticRegression
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import os

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plco_data_path",
        default="/scratch/project1/plco/lung_prsn.csv",
        help="Location of PLCO csv",
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate to use for SGD",
    )

    parser.add_argument(
        "--regularization_lambda",
        default=0,
        type=float,
        help="Weight to use for L2 regularization",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch_size to use for SGD"
    )

    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="number of epochs to use for training"
    )

    parser.add_argument(
        "--results_path",
        default="results.json",
        help="Where to save results"
    )

    return parser

def load_data(args: argparse.Namespace) -> tuple[list, list, list]:
    '''
    Load PLCO data from csv file and split into train validation and testing sets.
    '''
    reader = DictReader(open(args.plco_data_path,"r"))
    rows = [r for r in reader]
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)
    random.shuffle(rows)
    train, val, test = rows[:NUM_TRAIN], rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL], rows[NUM_TRAIN+NUM_VAL:]

    print(f"Data split: {len(train)} train, {len(val)} val, {len(test)} test samples")
    return train, val, test

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    print("Loading data from {}".format(args.plco_data_path))
    train, val, test = load_data(args)

    # TODO: Define feature configuration for your model
    # Example configurations:
    # For age-only model: feature_config = {"numerical": ["age"]}
    # For full model: feature_config = {
    #     "numerical": ["age", "pack_years"],  # Features to normalize
    #     "categorical": ["sex", "race7"],     # Features for one-hot encoding
    #     "ordinal": ["educat"]                # Features for integer encoding
    # }
    feature_config = None

    print("Initializing vectorizer and extracting features")
    # TODO: Implement a vectorizer to convert the questionnaire features into a feature vector
    plco_vectorizer = Vectorizer(feature_config)

    # TODO: Fit the vectorizer on the training data (i.e. compute means for normalization, etc)
    plco_vectorizer.fit(train)

    # TODO: Featurize the training, validation and testing data
    train_X = plco_vectorizer.transform(train)
    val_X = plco_vectorizer.transform(val)
    test_X = plco_vectorizer.transform(test)

    train_Y = np.array([int(r["lung_cancer"]) for r in train])
    val_Y = np.array([int(r["lung_cancer"]) for r in val])
    test_Y = np.array([int(r["lung_cancer"]) for r in test])

    print("Training model")

    # TODO: Initialize and train a logistic regression model
    model = LogisticRegression(
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        regularization_lambda=args.regularization_lambda, 
        verbose=True
    )

    model.fit(train_X, train_Y)

    print("Evaluating model")

    pred_train_Y = model.predict_proba(train_X)
    pred_val_Y = model.predict_proba(val_X)

    results = {
        "train_auc": roc_auc_score(train_Y, pred_train_Y),
        "val_auc": roc_auc_score(val_Y, pred_val_Y)
    }

    print(results)

    print("Saving results to {}".format(args.results_path))

    json.dump(results, open(args.results_path, "w"), indent=True, sort_keys=True)

    # Print AUC on validation set. Note, you always want to use validation set to tune your model.
    print(f"Validation AUC: {results['val_auc']}")

    

    # Compute AUC on test set and print for submission. Note, you should not use test set to tune your model.
    # Uncomment these lines only when you're ready for final evaluation:
    # pred_test_Y = model.predict_proba(test_X)
    # test_auc = roc_auc_score(test_Y, pred_test_Y)
    # print(f"Test AUC: {test_auc:.4f}")

    print("Done")

    return results

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)