import os
import re
import glob
from enum import Enum
from typing import Optional
from zipfile import ZipFile
from argparse import ArgumentParser, _SubParsersAction

import pandas as pd
import geopandas as gpd

from joblib import Parallel, delayed
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler

from infrastructure.logger import init_logger
from ml.preprocessing import ZonalProcessor, ZonalProcessorOptions, DownsampleEnum

# ----------------------------------------------------------------------------
#  CONVERT DARWINCORE TO CSV
# ----------------------------------------------------------------------------


class PreprocessGBIFCommandOptions(BaseModel):
    input_path: str
    output_path: str


class PreprocessGBIFCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PreprocessGBIFCommand")

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "occurence", help="Preprocess GBIF data to simple occurence data")
        parser.set_defaults(func=PreprocessGBIFCommand())
        parser.add_argument(
            "input_path",
            type=str,
            help="Path to DarwinCore ZIP containing the occurence dataset")
        parser.add_argument(
            "output_path",
            type=str,
            help="Output filename to summarized GBIF occurence data")

    def __call__(self, **kwargs) -> None:
        # parse args
        self.config = PreprocessGBIFCommandOptions(**kwargs)

        # read DarwinCore zip file
        with ZipFile(self.config.file_path, "r") as zip_ref:
            # read citations file
            with zip_ref.open("citations.txt") as citations_file:
                citations = citations_file.read().decode("utf-8")
                self.logger.info(citations)

            # read occurence file
            with zip_ref.open("occurrence.txt") as occurrence_file:
                # read occurrence file
                df = pd.read_csv(occurrence_file, sep="\t")
                df["ts"] = pd.to_datetime(df["eventDate"], format='ISO8601')

                # print min max ts
                self.logger.info(f"Min ts: {df['ts'].min()}")
                self.logger.info(f"Max ts: {df['ts'].max()}")

                # if decimalLatitude and decimalLongitude are null, try to use footprintWKT
                df["latitude"] = df.apply(
                    PreprocessGBIFCommand.coalesce_coordinate(
                        "decimalLatitude"),
                    axis=1)
                df["longitude"] = df.apply(
                    PreprocessGBIFCommand.coalesce_coordinate(
                        "decimalLongitude"),
                    axis=1)

                # subset dataset
                df_subset = df[[
                    "occurrenceID", "ts", "latitude", "longitude", "species"
                ]]

                # sort by ts descending
                df_subset = df_subset.sort_values("ts", ascending=False)

                # save to csv
                df_subset.to_csv(self.config.output_filename, index=False)
                self.logger.info(f"File saved to {self.config.output_filename}")

    @staticmethod
    def coalesce_coordinate(col: str) -> float:

        def proc(row):
            # if not null, return the value
            if not pd.isnull(row[col]):
                return float(row[col])

            # extract from WKT
            matches = re.findall(r"POINT\((.+) (.+)\)", row["footprintWKT"])
            if "lat" in col:
                return float(matches[0][0])

            return float(matches[0][1])

        return proc


# ----------------------------------------------------------------------------
#  CREATE ZONE ID
# ----------------------------------------------------------------------------


class PreprocessZoneIDCommandOptions(BaseModel):
    input_path: str
    output_path: str


class PreprocessZoneIDCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PreprocessZoneIDCommand")

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "zone-id",
            help="Adds zone id to existing zone polygon grid Shapefile")
        parser.set_defaults(func=PreprocessZoneIDCommand())
        parser.add_argument(
            "input_path",
            type=str,
            help="Path to Shapefile containing the grid or zone to calculate the zonal statistics from"
        )
        parser.add_argument(
            "output_path", type=str, help="Output Shapefile path")

    def __call__(self, args) -> None:
        # parse args
        self.config = PreprocessZoneIDCommandOptions(**vars(args))

        # read shapefile
        df_geom = gpd.read_file(self.config.input_path)

        # add sequential ZONE_ID attribute
        df_geom["ZONE_ID"] = range(1, len(df_geom) + 1)

        # save to shapefile
        df_geom.to_file(self.config.output_path, driver="ESRI Shapefile")


# ----------------------------------------------------------------------------
#  ZONAL STATISTICS
# ----------------------------------------------------------------------------


class PreprocessZonalStatsCommandOptions(BaseModel):
    raster_path: str
    zone_path: str
    occurence_path: str
    output_path: str
    downsample: DownsampleEnum = DownsampleEnum.NONE
    test_size: float = 0.2

    # job settings
    temp_dir: str = "./temp"
    jobs: int = 1


class PreprocessZonalStatsCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PreprocessCopernicusCommand")

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "zonal-stats",
            help="Preprocess GBIF occurence data and Copernicus Marine raster data to zonal statistics"
        )
        parser.set_defaults(func=PreprocessZonalStatsCommand())
        parser.add_argument(
            "occurence_path",
            type=str,
            help="Path to simple GBIF occurence data")
        parser.add_argument(
            "zone_path",
            type=str,
            help="Path to zone polygon Shapefile (it must have ZONE_ID in the attribute table)"
        )
        parser.add_argument(
            "raster_path",
            type=str,
            help="Path to a directory containing netCDF files")
        parser.add_argument(
            "output_path",
            type=str,
            help="Output path to save the zonal statistics dataset")

        parser.add_argument(
            "--downsample",
            type=str,
            choices=["none", "weekly", "monthly"],
            default="none",
            help="Time frequency to downsample to (default: none)")
        parser.add_argument(
            "--temp-dir",
            type=str,
            help="Path to temporary directory to store intermediate files")

    def __call__(self, args) -> None:
        self.config = PreprocessZonalStatsCommandOptions(**vars(args))

        # create temp dir
        os.makedirs(self.config.temp_dir, exist_ok=True)

        # find all raster files
        raster_files = glob.glob(f"{self.config.raster_path}/*.nc")
        self.logger.info("Found %d raster files", len(raster_files))

        # process each raster file
        jobs = [
            ZonalProcessorOptions(
                raster_file=f,
                occurence_file=self.config.occurence_path,
                zone_file=self.config.zone_path,
                output_path=self.config.temp_dir,
                downsample=self.config.downsample) for f in raster_files
        ]

        Parallel(
            n_jobs=self.config.jobs, verbose=1)(
                delayed(lambda x: ZonalProcessor(x)())(job) for job in jobs)
        self.logger.info("Finished processing zonal statistics")

        # merge all temp data to one dataframe
        self.merge()
        self.logger.info("Finished!")

    def merge(self):
        self.logger.info("Merging all variables...")

        # find all temp parquet
        temp_files = glob.glob(f"{self.config.temp_dir}/*.parquet")

        # first parquet as reference
        df_final = pd.read_parquet(temp_files[0])
        for f in temp_files[1:]:
            # read next dataset
            df_temp = pd.read_parquet(f)

            # merge on ZONE_ID
            df_final = df_final.merge(df_temp, on=["zone_id", "ts"]) \
                .drop(columns=["target_x"], errors="ignore")

        # drop extra merge fields
        df_final.drop(
            columns=[x for x in df_final.columns if x.startswith("target_")])

        # rearange columns
        cols = list(x for x in df_final.columns if not x.startswith("target_"))
        cols.remove("ts")
        cols.remove("target")
        cols.insert(1, "ts")
        cols.append("target")

        df_final = df_final[cols]

        # save merged dataset
        df_final.to_parquet(self.config.output_path, engine="pyarrow")


# ----------------------------------------------------------------------------
#  SPLIT DATASET
# ----------------------------------------------------------------------------


class PreprocessSplitDatasetCommandOptions(BaseModel):
    dataset_file: str
    output_path: str

    test_size: float = 0.2
    stratify: bool = True
    undersample: bool = False


class PreprocessSplitDatasetCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PreprocessSplitDatasetCommand")

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "split", help="Samples and split dataset into train and test sets")
        parser.set_defaults(func=PreprocessSplitDatasetCommand())
        parser.add_argument(
            "dataset_file",
            type=str,
            help="Path to Parquet file containing the full zonal statistics data"
        )
        parser.add_argument(
            "output_path",
            type=str,
            help="Output folder for the train and test split")

        parser.add_argument(
            "--test-size",
            type=float,
            default=0.2,
            help="Test size in proportion of the dataset (default: 0.2)")
        parser.add_argument(
            "--stratify",
            action='store_true',
            help="Perform stratified sampling (default: True)")
        parser.add_argument(
            "--undersample",
            action='store_true',
            help="Perform undersampling (default: False)")

    def __call__(self, args) -> None:
        self.config = PreprocessSplitDatasetCommandOptions(**vars(args))

        # load dataset
        self.logger.info("Loading dataset...")
        df = pd.read_parquet(self.config.dataset_file)
        self.logger.info("Dataset loaded! Rows: %d", len(df))

        # split features
        X, y = df.drop(columns=["target"]), df["target"]

        # should undersample?
        if self.config.undersample:
            rus = RandomUnderSampler(random_state=42)
            X, y = rus.fit_resample(X, y)

            self.logger.info("Dataset undersampling result: %d", len(X))

        # split dataset
        if self.config.stratify:
            self.logger.info("Using stratified sampling...")
            df_train, df_test = train_test_split(
                X.assign(target=y), test_size=self.config.test_size, stratify=y)
        else:
            self.logger.info("Not using stratified sampling...")
            df_train, df_test = train_test_split(
                X.assign(target=y), test_size=self.config.test_size)

        # save dataset
        self.logger.info("Saving dataset...")
        df_train.to_parquet(
            os.path.join(self.config.output_path, "train.parquet"),
            engine="pyarrow")
        df_test.to_parquet(
            os.path.join(self.config.output_path, "test.parquet"),
            engine="pyarrow")


# ----------------------------------------------------------------------------
#  FEATURE SELECTION
# ----------------------------------------------------------------------------


class FeatureSelectionStrategyEnum(str, Enum):
    TOPK = "topk"
    PERCENTILE = "percentile"
    MANUAL = "manual"


class PreprocessFeatureSelectionCommandOptions(BaseModel):
    dataset_file: str
    output_file: str

    topk: int = 10
    strategy: FeatureSelectionStrategyEnum
    features: Optional[str] = ""


class PreprocessFeatureSelectionCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PreprocessFeatureSelectionCommand")

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "feature_selection",
            help="Performs feature selection using information gain as the metric"
        )
        parser.set_defaults(func=PreprocessFeatureSelectionCommand())
        parser.add_argument(
            "dataset_file",
            type=str,
            help="Path to Parquet file containing the full zonal statistics data"
        )
        parser.add_argument(
            "output_file",
            type=str,
            help="Output file for the selected features")

        parser.add_argument(
            "--topk",
            type=int,
            default=10,
            help="Top-k number of features to select (default: 10)")
        parser.add_argument(
            "--strategy",
            choices=["topk", "percentile", "manual"],
            default="fpr",
            help="Feature selection strategy (default: percentile)")
        parser.add_argument(
            "--features",
            type=str,
            help="List of features to select (quote and separate by comma)")

    def __call__(self, args) -> None:
        self.config = PreprocessFeatureSelectionCommandOptions(**vars(args))

        # load dataset
        self.logger.info("Loading dataset...")
        df = pd.read_parquet(self.config.dataset_file)
        self.logger.info("Dataset loaded! Rows: %d", len(df))

        # split features
        X, y = df.drop(columns=["zone_id", "ts", "target"]), df["target"]
        self.logger.info("Input features count: %d", X.shape[1])

        # check strategy
        features = self.config.features
        if self.config.strategy == FeatureSelectionStrategyEnum.MANUAL:
            self.logger.info("Using MANUAL strategy...")
        elif self.config.strategy == FeatureSelectionStrategyEnum.PERCENTILE:
            self.logger.info("Using PERCENTILE strategy...")
            selector = SelectPercentile(
                mutual_info_classif, percentile=self.config.topk)
            _ = selector.fit_transform(X, y)

            features = selector.get_feature_names_out()
            self.logger.info("Scores: %s", selector.scores_)
        elif self.config.strategy == FeatureSelectionStrategyEnum.TOPK:
            self.logger.info("Using TOPK strategy...")
            selector = SelectKBest(mutual_info_classif, k=self.config.topk)
            _ = selector.fit_transform(X, y)

            features = selector.get_feature_names_out()
            self.logger.info("Scores: %s", selector.scores_)

        # print selected features
        self.logger.info("Selected features count: %d", len(features))
        self.logger.info("Selected features: %s", features)

        # apply feature selection
        all_features = ["zone_id", "ts"]
        all_features.extend(features)
        all_features.extend(["target"])

        # save dataset
        self.logger.info("Saving dataset...")
        df[all_features].to_parquet(self.config.output_file, engine="pyarrow")
