import argparse

import matplotlib
from pydantic import ValidationError

from commands import (
    DownloadCmemsCommand,
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
        type=int,
        default=1,
        help="Number of jobs to run in parallel",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=21,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="./tmp",
        help="Path to temporary directory to store intermediate files",
    )

    # add sub commands
    subparsers = parser.add_subparsers()

    # --- download command
    DownloadCmemsCommand.add_parser(subparsers)

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
    if "func" not in args:
        print("Use --help to show available commands")
        return

    # execute tool
    args.func(args)


if __name__ == "__main__":
    # change matplotlib backend
    matplotlib.use("Agg")

    try:
        # bootstrap
        main()
    except ValidationError as e:
        print(e)
        exit(-1)
