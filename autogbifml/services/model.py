import os
from enum import Enum
from typing import Any

import mlflow
import optuna
import joblib
from pydantic import BaseModel

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)


class AlgorithmEnum(Enum):
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"


def read_dataset(path: str):
    # load dataset
    df = pd.read_parquet(path)

    # create X and y
    X = df.drop(
        columns=["zone_id", "ts", "target", "country", "continent"], errors="ignore"
    )
    y = df["target"]

    return X, y


class Trainer:
    def __init__(self, algorithm: AlgorithmEnum, model_params: dict) -> None:
        self.algorithm = algorithm
        self.create_model(model_params)

    def fit(self, X, y):
        if self.model == None:
            raise ValueError("Model not created")

        self.model.fit(X, y)

    def predict(self, X):
        if self.model == None:
            raise ValueError("Model not created")

        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model == None:
            raise ValueError("Model not created")

        return self.model.predict_proba(X)

    def evaluate(self, X, y) -> dict:
        if self.model == None:
            raise ValueError("Model not created")

        y_pred = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "mcc": matthews_corrcoef(y, y_pred),
            "roc_auc": roc_auc_score(y, y_pred),
            "ap": average_precision_score(y, y_pred),
        }

    def create_model(self, params: dict):
        if self.algorithm == AlgorithmEnum.DECISION_TREE:
            self.model = DecisionTreeClassifier(**params)

        elif self.algorithm == AlgorithmEnum.RANDOM_FOREST:
            self.model = RandomForestClassifier(**params)

        elif self.algorithm == AlgorithmEnum.XGBOOST:
            self.model = XGBClassifier(**params)

        elif self.algorithm == AlgorithmEnum.CATBOOST:
            self.model = CatBoostClassifier(**params)

    def save(self, output_path: str):
        if self.algorithm == AlgorithmEnum.DECISION_TREE:
            joblib.dump(
                self.model, os.path.join(output_path, "decision_tree_model.joblib")
            )

        elif self.algorithm == AlgorithmEnum.RANDOM_FOREST:
            joblib.dump(
                self.model, os.path.join(output_path, "random_forest_model.joblib")
            )

        elif self.algorithm == AlgorithmEnum.XGBOOST:
            self.model.save_model(os.path.join(output_path, "xgboost_model.ubj"))

        elif self.algorithm == AlgorithmEnum.CATBOOST:
            self.model.save_model(os.path.join(output_path, "xgboost_model.cbm"))

    def load(self, input_path: str):
        if (
            self.algorithm == AlgorithmEnum.DECISION_TREE
            or self.algorithm == AlgorithmEnum.RANDOM_FOREST
        ):
            self.model = joblib.load(input_path)

        elif self.algorithm == AlgorithmEnum.XGBOOST:
            self.model = XGBClassifier()
            self.model.load_model(input_path)

        elif self.algorithm == AlgorithmEnum.CATBOOST:
            self.model = CatBoostClassifier()
            self.model.load_model(input_path)


class OptunaObjectiveOptions(BaseModel):
    algorithm: AlgorithmEnum
    dataset_file: str

    cv: int = 10
    jobs: int = 1
    shuffle: bool = True
    random_seed: int = 21


class OptunaObjective:
    def __init__(self, config: OptunaObjectiveOptions) -> None:
        self.config = config

    def __call__(self, trial: optuna.Trial) -> Any:
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create parameters
            trial_params = self.get_param_grid(trial)

            # log params to mlflow
            mlflow.log_params(trial_params)

            # create scores
            scores = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "mcc": [],
                "roc_auc": [],
                "ap": [],
            }

            # perform cross validation
            cv = StratifiedKFold(
                n_splits=self.config.cv,
                shuffle=self.config.shuffle,
                random_state=self.config.random_seed,
            )
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
                print(f"Training fold {fold_i + 1}")

                # split data
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                # fit model
                clf = Trainer(self.config.algorithm, trial_params)
                clf.fit(X_train, y_train)

                # run prediction
                y_pred = clf.predict(X_test)

                # log metrics
                scores["accuracy"].append(accuracy_score(y_test, y_pred))
                scores["precision"].append(precision_score(y_test, y_pred))
                scores["recall"].append(recall_score(y_test, y_pred))
                scores["f1"].append(f1_score(y_test, y_pred))
                scores["mcc"].append(matthews_corrcoef(y_test, y_pred))
                scores["roc_auc"].append(roc_auc_score(y_test, y_pred))
                scores["ap"].append(average_precision_score(y_test, y_pred))

                # plot confusion matrix
                fig = Figure()
                ax = fig.subplots()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                mlflow.log_figure(fig, f"confusion_matrix_fold_{fold_i + 1}.png")

            # log metrics to mlflow
            for metric_name, metric_values in scores.items():
                mlflow.log_metric(metric_name, np.mean(metric_values))

            # return MCC score to maximize
            return np.mean(scores["mcc"])

    def load_data(self) -> None:
        self.X, self.y = read_dataset(self.config.dataset_file)

    def get_param_grid(self, trial: optuna.Trial) -> dict:
        # positive-negative class ratio
        n_neg = self.y[self.y == 0].shape[0]
        n_pos = self.y[self.y == 1].shape[0]

        if self.config.algorithm == AlgorithmEnum.DECISION_TREE:
            # https://arxiv.org/pdf/1812.02207
            return {
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                # fixed parameters
                "class_weight": "balanced",
                "random_state": self.config.random_seed,
            }

        elif self.config.algorithm == AlgorithmEnum.RANDOM_FOREST:
            # https://arxiv.org/pdf/1804.03515 -> replacement = bootstrap, number of tree max 1000
            # https://jmlr.org/papers/volume18/17-269/17-269.pdf -> number of tree, 10-1000
            return {
                # same as decision tree
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                # random forest specific
                "n_estimators": trial.suggest_int("n_estimators", 10, 1000, step=10),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                # fixed parameters
                "n_jobs": self.config.jobs,
                "random_state": self.config.random_seed,
                "class_weight": "balanced",
            }

        elif self.config.algorithm == AlgorithmEnum.XGBOOST:
            # https://arxiv.org/pdf/2111.06924
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "n_estimators": trial.suggest_int("n_estimators", 10, 1000, step=10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 1.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
                # fixed parameters
                "device": "gpu",
                "nthread": self.config.jobs,
                "scale_pos_weight": n_neg / n_pos,
                "seed": self.config.random_seed,
            }

        elif self.config.algorithm == AlgorithmEnum.CATBOOST:
            return {
                "depth": trial.suggest_int("depth", 2, 16),
                "iterations": trial.suggest_int("iterations", 10, 1000, step=10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 1.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
                # fixed parameters
                "verbose": 0,
                "task_type": "CPU",
                "scale_pos_weight": n_neg / n_pos,
                "random_seed": self.config.random_seed,
            }
