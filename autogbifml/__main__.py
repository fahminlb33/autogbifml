import argparse

from pydantic import ValidationError
from commands import (
    DownloadCommand,
    PreprocessGBIFCommand,
    PreprocessZoneIDCommand,
    PreprocessZonalStatsCommand,
    TuneCommand,
    TrainCommand,
    EvaluateCommand,
    PredictCommand,
)


def main():
    # create root parser
    parser = argparse.ArgumentParser(prog="autogbifml")
    parser.add_argument("-j",
                        "--jobs",
                        default=1,
                        help="Number of jobs to run in parallel",
                        type=int)

    # add sub commands
    subparsers = parser.add_subparsers()

    # --- download command
    parser_download = subparsers.add_parser(
        "download", help="Downloads Copernicus Marine data")
    parser_download.set_defaults(func=DownloadCommand())
    parser_download.add_argument("config_file",
                                 type=str,
                                 help="Path to yaml config")

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
        help=
        "Path to Shapefile containing the grid or zone to calculate the zonal statistics from"
    )
    parser_gbif.add_argument("output_path",
                             type=str,
                             help="Output Shapefile path")

    # zonal statistics and occurence
    parser_copernicus = subparser_preprocess.add_parser(
        "zonal-stats",
        help=
        "Preprocess GBIF occurence data and Copernicus Marine raster data to zonal statistics"
    )
    parser_copernicus.set_defaults(func=PreprocessZonalStatsCommand())
    parser_copernicus.add_argument("occurence_path",
                                   type=str,
                                   help="Path to simple GBIF occurence data")
    parser_copernicus.add_argument(
        "zone_path",
        type=str,
        help=
        "Path to zone polygon Shapefile (it must have ZONE_ID in the attribute table)"
    )
    parser_copernicus.add_argument(
        "raster_path",
        type=str,
        help="Path to a directory containing netCDF files")
    parser_copernicus.add_argument(
        "output_path",
        type=str,
        help="Output path to save the zonal statistics dataset")

    # --- tune command
    parser_tune = subparsers.add_parser(
        "tune", help="Perform hyperparameter tuning using Optuna")
    parser_tune.set_defaults(func=TuneCommand())

    # --- train command
    parser_train = subparsers.add_parser(
        "train", help="Train a single model for GBIF occurrence data")
    parser_train.set_defaults(func=TrainCommand())

    # --- evaluate command
    parser_predict = subparsers.add_parser(
        "evaluate", help="Evaluate model classification performance")
    parser_predict.set_defaults(func=EvaluateCommand())

    # --- predict command
    parser_predict = subparsers.add_parser(
        "predict", help="Run predictions on a single model")
    parser_predict.set_defaults(func=PredictCommand())

    # parse CLI
    args = parser.parse_args()
    print(args)
    # args.func(args)


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        print(e)
    except Exception as e:
        print(e)
