import os
import time
import glob
from typing import Any
from argparse import ArgumentParser, _SubParsersAction

import yaml
from pydantic import BaseModel

from services.base import BaseCommand
from services.model import AlgorithmEnum, Trainer, read_dataset

# ----------------------------------------------------------------------------
#  TRAIN A MODEL
# ----------------------------------------------------------------------------


class TrainCommandOptions(BaseModel):
    dataset_path: str
    output_path: str

    algorithm: AlgorithmEnum
    params_file: str

    jobs: int = 1
    random_seed: int = 21


class TrainCommand(BaseCommand):
    def __init__(self) -> None:
        super(TrainCommand, self).__init__()

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
            "--params-file",
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

        # load dataset
        self.logger.info("Loading dataset...")
        X_train, y_train = read_dataset(train_file)
        X_test, y_test = read_dataset(test_file)

        # create trainer
        model = Trainer(
            algorithm=self.config.algorithm,
            model_params=yaml.safe_load(open(self.config.params_file, "r")) or {},
        )

        # fit model
        self.logger.info("Fitting model...")
        st = time.time()
        model.fit(X_train, y_train)
        dt = time.time() - st
        self.logger.info(f"Training duration: {dt:.5f}")

        # evaluate model
        self.logger.info("Evaluating model...")
        st = time.time()
        scores = model.evaluate(X_test, y_test)
        dt = time.time() - st
        self.logger.info(f"Evaluation duration: {dt:.5f}")

        self.logger.info(f"Evaluation scores:")
        for k, v in scores.items():
            self.logger.info(f"{k} = {v:.5f}")

        # create output dir
        os.makedirs(self.config.output_path, exist_ok=True)

        # save model
        model.save(self.config.output_path)
