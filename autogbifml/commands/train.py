import os
from typing import Any, Optional

import yaml
import optuna
import mlflow

from pydantic import BaseModel

from infrastructure.logger import init_logger
from ml.training import init_matplotlib, AlgorithmEnum, OptunaObjective, OptunaObjectiveOptions


class TuneCommandOptions(BaseModel):
    dataset_path: str
    output_path: str
    name: str
    algorithm: AlgorithmEnum

    trials: int = 100
    cv: int = 10
    shuffle: bool = True
    random_seed: int = 21
    db_path: Optional[str] = "tune.db"
    tracking_url: Optional[str] = None
    jobs: int = 1


class TuneCommand:

    def __init__(self) -> None:
        self.logger = init_logger("TuneCommand")
        init_matplotlib()

    def __call__(self, args) -> Any:
        # parse args
        self.config = TuneCommandOptions(**vars(args))

        # set mlflow tracking
        if self.config.tracking_url:
            self.logger.info(
                f"Setting mlflow tracking url: {self.config.tracking_url}")
            mlflow.set_tracking_uri(self.config.tracking_url)

        # create objective
        objective = OptunaObjective(
            OptunaObjectiveOptions(
                algorithm=self.config.algorithm,
                dataset_path=self.config.dataset_path,
                cv=self.config.cv,
                shuffle=self.config.shuffle,
                random_seed=self.config.random_seed,
                jobs=self.config.jobs))

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
            storage=f"sqlite:///{self.config.db_path}",
            load_if_exists=True)

        # start optimization
        self.logger.info(
            f"Starting optimization with {self.config.trials} trials...")
        study.optimize(objective, n_trials=self.config.trials)

        # get best parameters
        yaml.dump(
            study.best_params,
            open(
                os.path.join(self.config.output_path,
                             f"best_params_{self.config.algorithm.value}.yml"), "w"))

    def get_or_create_experiment(self, experiment_name):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            self.logger.info(f"Found experiment: {experiment.experiment_id}")
            return experiment.experiment_id
        else:
            self.logger.info(f"Creating experiment: {experiment_name}")
            return mlflow.create_experiment(experiment_name)


class TrainCommand:

    def __call__(self) -> Any:
        pass


class EvaluateCommand:

    def __call__(self) -> Any:
        pass
