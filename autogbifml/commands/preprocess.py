import os
import re
import glob
import shutil
from enum import Enum
from typing import Optional, Literal
from zipfile import ZipFile
from argparse import ArgumentParser, _SubParsersAction

import pandas as pd
import geopandas as gpd

from yaml import safe_load
from joblib import Parallel, delayed
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif

from services.base import BaseCommand
from services.preprocessing import ZonalRasterProcessor

# ----------------------------------------------------------------------------
#  CONVERT DARWINCORE TO CSV
# ----------------------------------------------------------------------------


class PreprocessGBIFCommandOptions(BaseModel):
    input_file: str
    output_file: str


class PreprocessGBIFCommand(BaseCommand):
    def __init__(self) -> None:
        super(PreprocessGBIFCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "occurrence", help="Preprocess GBIF data to simple occurence data"
        )

        parser.set_defaults(func=PreprocessGBIFCommand())
        parser.add_argument(
            "input_file",
            type=str,
            help="Path to DarwinCore ZIP containing the occurence dataset",
        )
        parser.add_argument(
            "output_file",
            type=str,
            help="Absolute path to summarized GBIF occurence data",
        )

    def __call__(self, args) -> None:
        # parse args
        self.config = PreprocessGBIFCommandOptions(**vars(args))

        # read DarwinCore zip file
        with ZipFile(self.config.input_file, "r") as zip_ref:
            # read citations file
            with zip_ref.open("citations.txt") as citations_file:
                citations = citations_file.read().decode("utf-8")
                self.logger.info(citations)

            # read occurence file
            with zip_ref.open("occurrence.txt") as occurrence_file:
                # read occurrence file
                df = pd.read_csv(occurrence_file, sep="\t")
                df["ts"] = pd.to_datetime(df["eventDate"], format="ISO8601")

                # print min max ts
                self.logger.info(f"Min ts: {df['ts'].min()}")
                self.logger.info(f"Max ts: {df['ts'].max()}")

                # if decimalLatitude and decimalLongitude are null, try to use footprintWKT
                df["latitude"] = df.apply(
                    PreprocessGBIFCommand.coalesce_coordinate("decimalLatitude"), axis=1
                )
                df["longitude"] = df.apply(
                    PreprocessGBIFCommand.coalesce_coordinate("decimalLongitude"),
                    axis=1,
                )

                # subset dataset
                df_subset = df[
                    ["occurrenceID", "ts", "latitude", "longitude", "species"]
                ]

                # sort by ts descending
                df_subset = df_subset.sort_values("ts", ascending=False)

                # save to csv
                df_subset.to_csv(self.config.output_file, index=False)
                self.logger.info(f"File saved to {self.config.output_file}")

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
    input_file: str
    output_file: str


class PreprocessZoneIDCommand(BaseCommand):
    def __init__(self) -> None:
        super(PreprocessZoneIDCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "zone-id", help="Adds zone id to existing zone polygon grid Shapefile"
        )

        parser.set_defaults(func=PreprocessZoneIDCommand())
        parser.add_argument(
            "input_file",
            type=str,
            help="Path to Shapefile containing the grid or zone to calculate the zonal statistics from",
        )
        parser.add_argument("output_file", type=str, help="Output Shapefile path")

    def __call__(self, args) -> None:
        # parse args
        self.config = PreprocessZoneIDCommandOptions(**vars(args))

        # read shapefile
        df_geom = gpd.read_file(self.config.input_file)

        # add sequential ZONE_ID attribute
        df_geom["ZONE_ID"] = range(1, len(df_geom) + 1)

        # save to shapefile
        df_geom.to_file(self.config.output_file, driver="ESRI Shapefile")


# ----------------------------------------------------------------------------
#  ZONAL STATISTICS
# ----------------------------------------------------------------------------


class PreprocessZonalVectorItem(BaseModel):
    type: Literal["grid-distance"]
    path: str
    zone_id: str
    select: list[str]


class PreprocessZonalRasterItem(BaseModel):
    type: Literal["gebco", "cmems"]
    path: str
    select: list[str]
    derive: list[Literal["sum", "mean", "count"]]


class PreprocessZonalProfile(BaseModel):
    output_file: str
    occurrence_file: str
    zones_file: str

    vectors: list[PreprocessZonalVectorItem]
    rasters: list[PreprocessZonalRasterItem]


class PreprocessZonalStatsCommandOptions(BaseModel):
    profile_file: str

    # job settings
    temp_dir: str = "./temp"
    jobs: int = 1


class PreprocessZonalStatsCommand(BaseCommand):
    def __init__(self) -> None:
        super(PreprocessZonalStatsCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "zonal-stats",
            help="Preprocess input data (vectors, rasters) for machine learning",
        )

        parser.set_defaults(func=PreprocessZonalStatsCommand())
        parser.add_argument(
            "profile_file", type=str, help="Path to data preprocessing profile"
        )

    def __call__(self, args) -> None:
        self.config = PreprocessZonalStatsCommandOptions(**vars(args))

        # load profile
        self.profile = PreprocessZonalProfile(
            **safe_load(open(self.config.profile_file, "r"))
        )
        self.logger.info("Loaded profile file!")

        # create temp dir
        temp_path = os.path.abspath(self.config.temp_dir)
        shutil.rmtree(temp_path, ignore_errors=True)
        os.makedirs(temp_path, exist_ok=True)

        # run processing
        self.process_rasters()
        self.process_vectors()

        # merge all temp data to one dataframe
        self.merge()
        self.logger.info("Finished!")

    def process_rasters(self):
        # create job config
        jobs = [
            {
                "type": f.type,
                "derived_vars": f.derive,
                "selected_vars": f.select,
                "raster_file": f.path,
                "occurrence_file": self.profile.occurrence_file,
                "zone_file": self.profile.zones_file,
                "output_path": self.config.temp_dir,
            }
            for f in self.profile.rasters
        ]

        # create parallel processor
        processor_job_fun = delayed(lambda x: ZonalRasterProcessor(**x)())
        processor_parallel = Parallel(n_jobs=self.config.jobs, verbose=1)

        # start jobs
        processor_parallel(processor_job_fun(job) for job in jobs)
        self.logger.info("Finished processing zonal statistics")

    def process_vectors(self):
        for vector_item in self.profile.vectors:
            # open vector
            gdf = gpd.read_file(vector_item.path)

            # subset and rename
            gdf_subset: gpd.GeoDataFrame = gdf[
                [vector_item.zone_id] + vector_item.select
            ]
            gdf_subset = gdf_subset.rename(columns={vector_item.zone_id: "zone_id"})

            # save to parquet
            gdf_subset.to_parquet(f"{self.config.temp_dir}/{vector_item.type}.parquet")

    def merge(self):
        self.logger.info("Merging all variables...")

        # find all temp parquet
        temp_files = glob.glob(f"{self.config.temp_dir}/*.parquet")

        # first parquet as reference
        df_final = pd.read_parquet(temp_files[0]).rename(
            columns={"target": "preserve_target"}
        )
        for f in temp_files[1:]:
            # read next dataset
            df_temp = pd.read_parquet(f)

            # merge on ZONE_ID
            if "cmems" in f:
                join_cols = ["zone_id", "ts"]
            else:
                join_cols = ["zone_id"]

            # merge
            df_final = df_final.merge(df_temp, on=join_cols).drop(
                columns=["target_x"], errors="ignore"
            )

        # drop extra merge fields
        df_final = df_final.drop(
            columns=[x for x in df_final.columns if x.startswith("target")]
        )

        # rename target col
        df_final = df_final.rename(columns={"preserve_target": "target"})

        # drop duplicates
        df_final = df_final.drop_duplicates(
            subset=["ts", "zone_id", "target"]
        )

        # rearange columns
        cols = list(
            x
            for x in df_final.columns
            if not x.startswith("target") and x not in ["ts", "zone_id"]
        )
        cols = ["ts", "zone_id", *cols, "target"]

        # save merged dataset
        df_final[cols].to_parquet(self.profile.output_file)


# ----------------------------------------------------------------------------
#  MERGE DATASET AND PERFORM SAMPLING
# ----------------------------------------------------------------------------


class PreprocessMergeDatasetOptions(BaseModel):
    dataset_path: str
    output_path: str

    test_size: float = 0.2
    random_seed: int = 42


class PreprocessMergeDatasetCommand(BaseCommand):
    def __init__(self) -> None:
        super(PreprocessMergeDatasetCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "merge", help="Samples and split dataset into train and test sets"
        )

        parser.set_defaults(func=PreprocessMergeDatasetCommand())
        parser.add_argument(
            "dataset_path",
            type=str,
            help="Path to directory  containing the full zonal statistics data asParquet file",
        )
        parser.add_argument(
            "output_path", type=str, help="Output folder for the train and test split"
        )

        parser.add_argument(
            "--test-size",
            type=float,
            default=0.2,
            help="Test size in proportion of the dataset (default: 0.2)",
        )

    def __call__(self, args) -> None:
        self.config = PreprocessMergeDatasetOptions(**vars(args))

        # load dataset
        self.logger.info("Merging all variables...")
        data_files = glob.glob(f"{self.config.dataset_path}/*.parquet")

        # first parquet as reference
        # min_samples = float("inf")
        all_dfs = []
        for f in data_files:
            # get country from filename
            country = (
                os.path.splitext(os.path.basename(f))[0]
                .replace("_1", "")
                .replace("_2", "")
            )

            # read next dataset
            df_temp = pd.read_parquet(f).assign(country=country)
            all_dfs.append(df_temp)

            self.logger.info(
                "Found %d rows in %s. Proportions: %s",
                df_temp.shape[0],
                country,
                df_temp["target"].value_counts(),
            )

        # concat all
        df_final = pd.concat(all_dfs, ignore_index=True)

        # drop duplicates
        df_final = df_final.drop_duplicates(
            subset=["ts", "zone_id", "country", "target"]
        )

        # print stats
        self.logger.info("Columns: %s", df_final.columns.values)
        self.logger.info(
            "Statistics: %s", df_final[["country", "target"]].value_counts()
        )

        # split dataset
        self.logger.info("Using stratified sampling...")
        df_train, df_test = train_test_split(
            df_final,
            test_size=self.config.test_size,
            stratify=df_final[["country", "target"]],
            random_state=self.config.random_seed,
        )

        # split statistics
        self.logger.info("Splitting statistics...")
        self.logger.info("TRAIN %s", df_train[["country", "target"]].value_counts())
        self.logger.info("TEST %s", df_test[["country", "target"]].value_counts())

        # save dataset
        self.logger.info("Saving dataset...")
        df_train.to_parquet(os.path.join(self.config.output_path, "train.parquet"))
        df_test.to_parquet(os.path.join(self.config.output_path, "test.parquet"))


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


class PreprocessFeatureSelectionCommand(BaseCommand):
    def __init__(self) -> None:
        super(PreprocessFeatureSelectionCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "feature-selection",
            help="Performs feature selection using information gain as the metric",
        )

        parser.set_defaults(func=PreprocessFeatureSelectionCommand())
        parser.add_argument(
            "dataset_file",
            type=str,
            help="Path to Parquet file containing the full zonal statistics data",
        )
        parser.add_argument(
            "output_file", type=str, help="Output file for the selected features"
        )

        parser.add_argument(
            "--topk",
            type=int,
            default=10,
            help="Top-k number of features to select (default: 10)",
        )
        parser.add_argument(
            "--strategy",
            choices=["topk", "percentile", "manual"],
            default="topk",
            help="Feature selection strategy (default: topk)",
        )
        parser.add_argument(
            "--features",
            type=str,
            help="List of features to select (quote and separate by comma)",
        )

    def __call__(self, args) -> None:
        self.config = PreprocessFeatureSelectionCommandOptions(**vars(args))

        # load dataset
        self.logger.info("Loading dataset...")
        df = pd.read_parquet(self.config.dataset_file)
        self.logger.info("Dataset loaded! Rows: %d", len(df))

        # split features
        X, y = df.drop(columns=["zone_id", "ts", "target", "country"]), df["target"]
        self.logger.info("Input features count: %d", X.shape[1])

        # check strategy
        features = []
        if self.config.strategy == FeatureSelectionStrategyEnum.MANUAL:
            self.logger.info("Using MANUAL strategy...")
            features = self.config.features.split(",")
        elif self.config.strategy == FeatureSelectionStrategyEnum.PERCENTILE:
            self.logger.info("Using PERCENTILE strategy...")
            selector = SelectPercentile(
                mutual_info_classif, percentile=self.config.topk
            )
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
        all_features = ["zone_id"]
        all_features.extend(features)
        all_features.extend(["target"])

        # save dataset
        self.logger.info("Saving dataset...")
        df[all_features].to_parquet(self.config.output_file)
