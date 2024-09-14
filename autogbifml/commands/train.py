import os
import glob
from typing import Any, Optional
from argparse import ArgumentParser, _SubParsersAction

import yaml
import optuna
import mlflow

from pydantic import BaseModel

from infrastructure.logger import init_logger
from ml.training import (
    init_matplotlib,
    AlgorithmEnum,
    OptunaObjective,
    OptunaObjectiveOptions,
    DataLoader,
    Trainer,
)

# ----------------------------------------------------------------------------
#  PERFORMS HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------


class TuneCommandOptions(BaseModel):
    dataset_file: str
    output_path: str
    name: str
    algorithm: AlgorithmEnum

    trials: int = 100
    cv: int = 10
    shuffle: bool = True
    random_seed: int = 21
    storage: Optional[str] = "tune.db"
    tracking_url: Optional[str] = None
    jobs: int = 1


class TuneCommand:
    def __init__(self) -> None:
        self.logger = init_logger("TuneCommand")
        init_matplotlib()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "tune", help="Perform hyperparameter tuning using Optuna"
        )
        parser.set_defaults(func=TuneCommand())
        parser.add_argument(
            "dataset_file", type=str, help="Path to a training dataset file"
        )
        parser.add_argument(
            "output_path",
            type=str,
            help="Path to store the trained model and parameters",
        )

        parser.add_argument(
            "--name",
            type=str,
            required=True,
            help="Tuning study name (in optuna and Mlflow)",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            required=True,
            choices=["xgboost", "catboost", "random_forest", "decision_tree"],
            help="Algorithm to tune",
        )
        parser.add_argument(
            "--trials", type=int, default=100, help="Number of trials to run"
        )
        parser.add_argument(
            "--cv", type=int, default=10, help="Number of k in cross-validation"
        )
        parser.add_argument(
            "--shuffle", action="store_true", help="Shuffle the dataset before split"
        )
        parser.add_argument(
            "--storage",
            type=str,
            default="sqlite:///tune.db",
            help="Path to store optuna study database",
        )
        parser.add_argument(
            "--tracking-url",
            type=str,
            help="Absolute URI to Mlflow server for model tracking",
        )

    def __call__(self, args) -> Any:
        # parse args
        self.config = TuneCommandOptions(**vars(args))

        # create output dir
        os.makedirs(self.config.output_path, exist_ok=True)

        # set mlflow tracking
        if self.config.tracking_url:
            self.logger.info(f"Setting mlflow tracking url: {self.config.tracking_url}")
            mlflow.set_tracking_uri(self.config.tracking_url)

        # create objective
        objective = OptunaObjective(
            OptunaObjectiveOptions(
                algorithm=self.config.algorithm,
                dataset_file=self.config.dataset_file,
                cv=self.config.cv,
                shuffle=self.config.shuffle,
                random_seed=self.config.random_seed,
                jobs=self.config.jobs,
            )
        )

        # load dataset
        self.logger.info("Loading dataset...")
        objective.load_data()

        # create mlflow experiment
        experiment_id = self.get_or_create_experiment(self.config.name)
        mlflow.set_experiment(experiment_id=experiment_id)

        # create study
        study = optuna.create_study(
            direction="maximize",
            study_name=self.config.name,
            storage=self.config.storage,
            load_if_exists=True,
        )

        # start optimization
        self.logger.info(f"Starting optimization with {self.config.trials} trials...")
        study.optimize(objective, n_trials=self.config.trials)

        # get best parameters
        yaml.dump(
            study.best_params,
            open(
                os.path.join(
                    self.config.output_path,
                    f"best_params_{self.config.algorithm.value}.yml",
                ),
                "w",
            ),
        )

    def get_or_create_experiment(self, experiment_name):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            self.logger.info(f"Found experiment: {experiment.experiment_id}")
            return experiment.experiment_id
        else:
            self.logger.info(f"Creating experiment: {experiment_name}")
            return mlflow.create_experiment(experiment_name)


# ----------------------------------------------------------------------------
#  TRAIN A MODEL
# ----------------------------------------------------------------------------


class TrainCommandOptions(BaseModel):
    dataset_path: str
    output_path: str

    algorithm: AlgorithmEnum
    params_path: str

    jobs: int = 1
    random_seed: int = 21


class TrainCommand:
    def __init__(self) -> None:
        self.logger = init_logger("TrainCommand")
        init_matplotlib()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "train", help="Train a single model for GBIF occurrence data"
        )
        parser.set_defaults(func=TrainCommand())
        parser.add_argument(
            "dataset_path",
            type=str,
            help="Path to a training dataset directory containing train and test set",
        )
        parser.add_argument(
            "output_path",
            type=str,
            help="Path to store the trained model and parameters",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            required=True,
            choices=["xgboost", "catboost", "random_forest", "decision_tree"],
            help="Algorithm to train",
        )
        parser.add_argument(
            "--params-path",
            type=str,
            required=True,
            help="Path to model paramters YAML file",
        )

    def __call__(self, args) -> Any:
        self.config = TrainCommandOptions(**vars(args))

        # find dataset
        parquet_files = glob.glob(f"{self.config.dataset_path}/*.parquet")
        train_file = list(filter(lambda x: "train" in x, parquet_files))[0]
        test_file = list(filter(lambda x: "test" in x, parquet_files))[0]

        self.logger.info(f"Found train dataset: {train_file}")
        self.logger.info(f"Found test dataset: {test_file}")

        # create data loader
        loader = DataLoader()

        # load dataset
        self.logger.info("Loading dataset...")
        X_train, y_train = loader.read_parquet(train_file)
        X_test, y_test = loader.read_parquet(test_file)

        cols = list(X_train.columns)

        # preprocess
        self.logger.info("Preprocessing dataset...")
        X_train = loader.fit_transform(X_train)
        X_test = loader.transform(X_test)

        # create trainer
        model = Trainer(
            algorithm=self.config.algorithm,
            model_params=yaml.safe_load(open(self.config.params_path, "r")) or {},
        )

        # fit model
        self.logger.info("Fitting model...")
        model.fit(X_train, y_train)

        # evaluate model
        scores = model.evaluate(X_test, y_test)
        scores = {key: round(value, 2) for key, value in scores.items()}
        self.logger.info("Evaluation scores:\n%s", scores)

        # create output dir
        os.makedirs(self.config.output_path, exist_ok=True)

        # save loader
        loader.save(os.path.join(self.config.output_path, "loader.joblib"))

        # save model
        model.save(self.config.output_path)

        # save feature importance
        model.feature_importance(cols).to_csv(
            os.path.join(self.config.output_path, "importance.csv")
        )
