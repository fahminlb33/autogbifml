import argparse

from pydantic import ValidationError
from commands import (
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
        "-j", "--jobs", default=1, help="Number of jobs to run in parallel", type=int
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        default=21,
        help="Random seed for reproducibility",
        type=int,
    )

    # add sub commands
    subparsers = parser.add_subparsers()

    # --- preprocessing commands
    parser_preprocess = subparsers.add_parser(
        "preprocess", help="Preprocess GBIF and Copernicus Marine data"
    )
    subparser_preprocess = parser_preprocess.add_subparsers()

    PreprocessGBIFCommand.add_parser(subparser_preprocess)
    PreprocessZoneIDCommand.add_parser(subparser_preprocess)
    PreprocessZonalStatsCommand.add_parser(subparser_preprocess)
    PreprocessMergeDatasetCommand.add_parser(subparser_preprocess)
    PreprocessFeatureSelectionCommand.add_parser(subparser_preprocess)

    # --- tune command
    TuneCommand.add_parser(subparsers)

    # --- train command
    TrainCommand.add_parser(subparsers)

    # --- predict command
    PredictCommand.add_parser(subparsers)

    # parse CLI
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
