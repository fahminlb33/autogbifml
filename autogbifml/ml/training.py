import os
from enum import Enum
from typing import Any

from sklearnex import patch_sklearn

patch_sklearn()

import mlflow
import optuna
import joblib
from pydantic import BaseModel

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from infrastructure.logger import init_logger


def init_matplotlib():
    # change matplotlib backend
    matplotlib.use("Agg")


class AlgorithmEnum(Enum):
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"


class DataLoader:
    def read_parquet(self, path: str):
        # load dataset
        df = pd.read_parquet(path)

        # create X and y
        return df.drop(
            columns=["zone_id", "ts", "target", "country"], errors="ignore"
        ), df["target"]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numerical_encoder", MinMaxScaler(), [col for col in X.columns]),
            ]
        )

        return self.preprocessor.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(X)

    def load(self, input_path: str):
        self.preprocessor = joblib.load(input_path)

    def save(self, output_path: str):
        joblib.dump(self.preprocessor, output_path)


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

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        return (
            pd.Series(self.model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "variable", 0: "value"})
        )


class OptunaObjectiveOptions(BaseModel):
    algorithm: AlgorithmEnum
    dataset_file: str

    jobs: int = 1
    cv: int = 10
    shuffle: bool = True
    random_seed: int = 21


class OptunaObjective:
    def __init__(self, config: OptunaObjectiveOptions) -> None:
        self.config = config
        self.logger = init_logger("OptunaObjective")
        self.loader = DataLoader()

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
            }

            # perform cross validation
            cv = KFold(
                n_splits=self.config.cv,
                shuffle=self.config.shuffle,
                random_state=self.config.random_seed,
            )
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
                self.logger.info("Training fold %d", fold_i + 1)

                # split data
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                # preprocess X
                X_train = self.loader.fit_transform(X_train)
                X_test = self.loader.transform(X_test)

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

                # feature importance
                mlflow.log_text(
                    clf.feature_importance(self.X.columns).to_csv(),
                    f"feature_importance_fold_{fold_i + 1}.csv",
                )

                # plot confusion matrix
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                mlflow.log_figure(fig, f"confusion_matrix_fold_{fold_i + 1}.png")

                # plot ROC
                fig, ax = plt.subplots()
                RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax)
                mlflow.log_figure(fig, f"roc_curve_fold_{fold_i + 1}.png")

                # plot precision-recall
                fig, ax = plt.subplots()
                PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=ax)
                mlflow.log_figure(fig, f"precision_recall_fold_{fold_i + 1}.png")

                # close all matplotlib windows
                plt.close("all")

            # log metrics to mlflow
            for metric_name, metric_values in scores.items():
                mlflow.log_metric(metric_name, np.mean(metric_values))

            # return MCC score to maximize
            return np.mean(scores["mcc"])

    def load_data(self) -> None:
        self.X, self.y = self.loader.read_parquet(self.config.dataset_file)

    def get_param_grid(self, trial: optuna.Trial) -> dict:
        # positive-negative class ratio
        n_neg = self.y[self.y == 0].shape[0]
        n_pos = self.y[self.y == 1].shape[0]

        if self.config.algorithm == AlgorithmEnum.DECISION_TREE:
            return {
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "max_depth": trial.suggest_int("max_depth", 1, 100),
                "max_features": trial.suggest_categorical(
                    "max_features", [None, "sqrt"]
                ),
                "min_samples_split": trial.suggest_categorical(
                    "min_samples_split", [2, 5, 10]
                ),
                "min_samples_leaf": trial.suggest_categorical(
                    "min_samples_leaf", [1, 2, 4]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 3, 26),
                # fixed parameters
                "random_state": self.config.random_seed,
                "class_weight": "balanced",
            }

        elif self.config.algorithm == AlgorithmEnum.RANDOM_FOREST:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=10),
                "max_depth": trial.suggest_int("max_depth", 3, 100),
                "min_samples_split": trial.suggest_categorical(
                    "min_samples_split", [2, 5, 10]
                ),
                "min_samples_leaf": trial.suggest_categorical(
                    "min_samples_leaf", [1, 2, 4]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "max_features": trial.suggest_categorical(
                    "max_features_2", [None, "sqrt"]
                ),
                # fixed parameters
                "n_jobs": self.config.jobs,
                "random_state": self.config.random_seed,
                "class_weight": "balanced",
            }

        elif self.config.algorithm == AlgorithmEnum.XGBOOST:
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 110),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.1, 1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=10),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", 0.01, 1.0, log=True
                ),
                # fixed parameters
                "scale_pos_weight": n_neg / n_pos,
                "random_state": self.config.random_seed,
            }

        elif self.config.algorithm == AlgorithmEnum.CATBOOST:
            return {
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
                ),
                "iterations": trial.suggest_int("iterations", 10, 2000, step=10),
                "depth": trial.suggest_int("depth", 1, 16),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.1, 1, log=True),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", 0.01, 1.0, log=True
                ),
                # fixed parameters
                "verbose": 0,
                "task_type": "CPU",
                "bootstrap_type": "Bernoulli",
                "scale_pos_weight": n_neg / n_pos,
                "random_seed": self.config.random_seed,
            }
