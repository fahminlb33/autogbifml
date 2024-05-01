import argparse

from pydantic import ValidationError
from commands import (
    DownloadCommand,
    PreprocessGBIFCommand,
    PreprocessZoneIDCommand,
    PreprocessZonalStatsCommand,
    PreprocessSplitDatasetCommand,
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

    # gbif
    parser_gbif = subparser_preprocess.add_parser(
        "occurence", help="Preprocess GBIF data to simple occurence data")
    parser_gbif.set_defaults(func=PreprocessGBIFCommand())
    parser_gbif.add_argument(
        "input_path",
        type=str,
        help="Path to DarwinCore ZIP containing the occurence dataset")
    parser_gbif.add_argument(
        "output_path",
        type=str,
        help="Output filename to summarized GBIF occurence data")

    # zone-id
    parser_gbif = subparser_preprocess.add_parser(
        "zone-id", help="Adds zone id to existing zone polygon grid Shapefile")
    parser_gbif.set_defaults(func=PreprocessZoneIDCommand())
    parser_gbif.add_argument(
        "input_path",
        type=str,
        help="Path to Shapefile containing the grid or zone to calculate the zonal statistics from"
    )
    parser_gbif.add_argument(
        "output_path", type=str, help="Output Shapefile path")

    # zonal statistics and occurence
    parser_copernicus = subparser_preprocess.add_parser(
        "zonal-stats",
        help="Preprocess GBIF occurence data and Copernicus Marine raster data to zonal statistics"
    )
    parser_copernicus.set_defaults(func=PreprocessZonalStatsCommand())
    parser_copernicus.add_argument(
        "occurence_path", type=str, help="Path to simple GBIF occurence data")
    parser_copernicus.add_argument(
        "zone_path",
        type=str,
        help="Path to zone polygon Shapefile (it must have ZONE_ID in the attribute table)"
    )
    parser_copernicus.add_argument(
        "raster_path",
        type=str,
        help="Path to a directory containing netCDF files")
    parser_copernicus.add_argument(
        "output_path",
        type=str,
        help="Output path to save the zonal statistics dataset")

    parser_copernicus.add_argument(
        "--downsample",
        type=str,
        choices=["none", "weekly", "monthly"],
        default="none",
        help="Time frequency to downsample to (default: none)")
    parser_copernicus.add_argument(
        "--temp-dir",
        type=str,
        help="Path to temporary directory to store intermediate files")

    # split dataset
    parser_split = subparser_preprocess.add_parser(
        "split", help="Samples and split dataset into train and test sets")
    parser_split.set_defaults(func=PreprocessSplitDatasetCommand())
    parser_split.add_argument(
        "dataset_file",
        type=str,
        help="Path to Parquet file containing the full zonal statistics data")
    parser_split.add_argument(
        "output_path",
        type=str,
        help="Output folder for the train and test split")

    parser_split.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size in proportion of the dataset (default: 0.2)")
    parser_split.add_argument(
        "--stratify",
        action='store_true',
        help="Perform stratified sampling (default: True)")
    parser_split.add_argument(
        "--undersample",
        action='store_true',
        help="Perform undersampling (default: False)")

    # --- tune command
    parser_tune = subparsers.add_parser(
        "tune", help="Perform hyperparameter tuning using Optuna")
    parser_tune.set_defaults(func=TuneCommand())
    parser_tune.add_argument(
        "dataset_path", type=str, help="Path to a training dataset file")
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
        "--db-path",
        type=str,
        default="tune.db",
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
        "output_file",
        type=str,
        help="Path to store the predicted shapefile")
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
        help="Path to saved model file"
    )
    parser_predict.add_argument(
        "--saved-loader-file",
        type=str,
        required=True,
        help="Path to saved loader file"
    )
    parser_predict.add_argument(
        "--polygon-file",
        type=str,
        required=True,
        help="Path to zone polygon file"
    )
    parser_predict.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="Path to a dataset file to predict"
    )

    # parse CLI
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
