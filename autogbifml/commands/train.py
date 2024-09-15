import os
import glob
from typing import Any
from argparse import ArgumentParser, _SubParsersAction

import yaml
from pydantic import BaseModel

from services.base import BaseCommand
from services.model import AlgorithmEnum, DataLoader, Trainer

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
            model_params=yaml.safe_load(open(self.config.params_file, "r")) or {},
        )

        # fit model
        self.logger.info("Fitting model...")
        model.fit(X_train, y_train)

        # evaluate model
        scores = model.evaluate(X_test, y_test)
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
