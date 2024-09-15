import os
from typing import Any, Optional
from argparse import ArgumentParser, _SubParsersAction

import yaml
import optuna
import mlflow

from pydantic import BaseModel

from services.base import BaseCommand
from services.model import (
    AlgorithmEnum,
    OptunaObjective,
    OptunaObjectiveOptions,
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


class TuneCommand(BaseCommand):
    def __init__(self) -> None:
        super(TuneCommand, self).__init__()

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
