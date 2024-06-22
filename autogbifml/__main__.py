import argparse

from pydantic import ValidationError
from commands import (
    DownloadCommand,
    PreprocessGBIFCommand,
    PreprocessZoneIDCommand,
    PreprocessZonalStatsCommand,
    PreprocessMergeDatasetCommand,
    PreprocessFeatureSelectionCommand,
    TuneCommand,
    TrainCommand,
    PredictCommand,
)


def main():
    # create root parser
    parser = argparse.ArgumentParser(prog="autogbifml")
    parser.add_argument(
        "-j",
        "--jobs",
        default=1,
        help="Number of jobs to run in parallel",
        type=int)
    parser.add_argument(
        "-s",
        "--random-seed",
        default=21,
        help="Random seed for reproducibility",
        type=int)

    # add sub commands
    subparsers = parser.add_subparsers()

    # --- download command
    parser_download = subparsers.add_parser(
        "download", help="Downloads Copernicus Marine data")
    parser_download.set_defaults(func=DownloadCommand())
    parser_download.add_argument(
        "config_file", type=str, help="Path to yaml config")

    # --- preprocessing commands
    parser_preprocess = subparsers.add_parser(
        "preprocess", help="Preprocess GBIF and Copernicus Marine data")
    subparser_preprocess = parser_preprocess.add_subparsers()

    PreprocessGBIFCommand.add_parser(subparser_preprocess)
    PreprocessZoneIDCommand.add_parser(subparser_preprocess)
    PreprocessZonalStatsCommand.add_parser(subparser_preprocess)
    PreprocessMergeDatasetCommand.add_parser(subparser_preprocess)
    PreprocessFeatureSelectionCommand.add_parser(subparser_preprocess)

    # --- tune command
    parser_tune = subparsers.add_parser(
        "tune", help="Perform hyperparameter tuning using Optuna")
    parser_tune.set_defaults(func=TuneCommand())
    parser_tune.add_argument(
        "dataset_file", type=str, help="Path to a training dataset file")
    parser_tune.add_argument(
        "output_path",
        type=str,
        help="Path to store the trained model and parameters")

    parser_tune.add_argument(
        "--name",
        type=str,
        required=True,
        help="Tuning study name (in optuna and Mlflow)")
    parser_tune.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["xgboost", "catboost", "random_forest", "decision_tree"],
        help="Algorithm to tune")
    parser_tune.add_argument(
        "--trials", type=int, default=100, help="Number of trials to run")
    parser_tune.add_argument(
        "--cv", type=int, default=10, help="Number of k in cross-validation")
    parser_tune.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before split")
    parser_tune.add_argument(
        "--storage",
        type=str,
        default="sqlite:///tune.db",
        help="Path to store optuna study database")
    parser_tune.add_argument(
        "--tracking-url",
        type=str,
        help="Absolute URI to Mlflow server for model tracking")

    # --- train command
    parser_train = subparsers.add_parser(
        "train", help="Train a single model for GBIF occurrence data")
    parser_train.set_defaults(func=TrainCommand())
    parser_train.add_argument(
        "dataset_path",
        type=str,
        help="Path to a training dataset directory containing train and test set"
    )
    parser_train.add_argument(
        "output_path",
        type=str,
        help="Path to store the trained model and parameters")
    parser_train.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["xgboost", "catboost", "random_forest", "decision_tree"],
        help="Algorithm to train")
    parser_train.add_argument(
        "--params-path",
        type=str,
        required=True,
        help="Path to model paramters YAML file")

    # --- predict command
    parser_predict = subparsers.add_parser(
        "predict", help="Run predictions on a single model")
    parser_predict.set_defaults(func=PredictCommand())
    parser_predict.add_argument(
        "output_file", type=str, help="Path to store the predicted shapefile")
    parser_predict.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["xgboost", "catboost", "random_forest", "decision_tree"],
        help="Algorithm used to train the model")
    parser_predict.add_argument(
        "--saved-model-file",
        type=str,
        required=True,
        help="Path to saved model file")
    parser_predict.add_argument(
        "--saved-loader-file",
        type=str,
        required=True,
        help="Path to saved loader file")
    parser_predict.add_argument(
        "--polygon-file",
        type=str,
        required=True,
        help="Path to zone polygon file")
    parser_predict.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="Path to a dataset file to predict")

    # parse CLI
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
