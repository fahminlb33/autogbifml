from typing import Any
from argparse import ArgumentParser, _SubParsersAction

import pandas as pd
import geopandas as gpd

from pydantic import BaseModel

from services.base import BaseCommand
from services.model import AlgorithmEnum, Trainer, read_dataset

# ----------------------------------------------------------------------------
#  PERFORMS INFERENCE USING A PRETRAINED MODEL
# ----------------------------------------------------------------------------


class PredictCommandOptions(BaseModel):
    classifier_file: str
    loader_file: str
    polygon_file: str
    dataset_file: str
    output_file: str
    algorithm: AlgorithmEnum

    jobs: int = 1

# FIXME: this is not tested

class PredictCommand(BaseCommand):
    def __init__(self) -> None:
        super(PredictCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser("predict", help="Run predictions")

        parser.set_defaults(func=PredictCommand())
        parser.add_argument(
            "--algorithm",
            type=str,
            required=True,
            choices=["xgboost", "catboost", "random_forest", "decision_tree"],
            help="Algorithm used to train the model",
        )
        parser.add_argument(
            "--loader-file",
            type=str,
            required=True,
            help="Path to saved loader file",
        )
        parser.add_argument(
            "--classifier-file",
            type=str,
            required=True,
            help="Path to saved model file",
        )
        parser.add_argument(
            "--polygon-file", type=str, required=True, help="Path to zone polygon file"
        )
        parser.add_argument(
            "--dataset-file",
            type=str,
            required=True,
            help="Path to a dataset file to predict",
        )
        parser.add_argument(
            "--output-file",
            type=str,
            required=True,
            help="Path to store the predicted GeoJSON",
        )

    def __call__(self, args) -> Any:
        # parse args
        self.config = PredictCommandOptions(**vars(args))

        # create model
        self.logger.info("Loading prediction model...")
        model = Trainer(self.config.algorithm, model_params={})
        model.load(self.config.classifier_file)

        # load dataset
        self.logger.info("Loading dataset...")
        df_zonal = pd.read_parquet(self.config.dataset_file)
        print(df_zonal.info())

        # check if dataset has zone_id
        if "zone_id" not in df_zonal.columns:
            raise ValueError("zone_id column not found in zonal dataset")

        # load polygon dataset
        self.logger.info("Loading polygon...")
        df_geom: gpd.GeoDataFrame = gpd.read_file(self.config.polygon_file)

        # check if polygon has ZONE_ID
        if "ZONE_ID" not in df_geom.columns:
            raise ValueError("zone_id column not found in polygon dataset")

        # run prediction
        self.logger.info("Starting prediction...")

        # derive features
        X = df_zonal.drop(columns=["zone_id", "ts", "target"], errors="ignore")
        # X = loader.transform(df_zonal)

        # run predictions
        df_zonal_pred = df_zonal.copy()
        df_zonal_pred["predicted"] = model.predict(X)

        # construct prediction dataframe
        df_geom_pred = df_geom[["ZONE_ID", "geometry"]]
        df_geom_pred = (
            df_geom_pred.merge(
                df_zonal_pred, left_on="ZONE_ID", right_index=True, how="left"
            )
            .fillna(0)
            .drop(columns=["ZONE_ID"])
        )
        print(df_geom_pred.info())

        # save to shapefile
        self.logger.info("Saving polygon...")
        df_geom_pred.to_file(self.config.output_file, driver="GeoJSON")
