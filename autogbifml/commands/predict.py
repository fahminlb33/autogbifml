import os
import glob
from typing import Any, Optional

import yaml
import optuna
import mlflow

import pandas as pd
import geopandas as gpd

from pydantic import BaseModel

from infrastructure.logger import init_logger
from ml.training import (init_matplotlib, AlgorithmEnum, OptunaObjective,
                         OptunaObjectiveOptions, DataLoader, Trainer)


class PredictCommandOptions(BaseModel):
    saved_model_file: str
    saved_loader_file: str
    polygon_file: str
    dataset_file: str
    output_file: str
    algorithm: AlgorithmEnum

    jobs: int = 1

class PredictCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PredictCommand")
        init_matplotlib()

    def __call__(self, args) -> Any:
        # parse args
        self.config = PredictCommandOptions(**vars(args))

        # create data loader
        self.logger.info("Loading preprocessor...")
        loader = DataLoader()
        loader.load(self.config.saved_loader_file)

        # create model
        self.logger.info("Loading prediction model...")
        model = Trainer(self.config.algorithm, model_params={})
        model.load(self.config.saved_model_file)

        # load dataset
        self.logger.info("Loading dataset...")
        df = pd.read_parquet(self.config.dataset_file)

        # load polygon dataset
        self.logger.info("Loading polygon...")
        df_geom: gpd.GeoDataFrame = gpd.read_file(self.config.polygon_file)

        # check if polygon has zone_id
        if "ZONE_ID" not in df_geom.columns:
            raise ValueError("zone_id column not found in polygon dataset")

        # get unique time steps
        ts = df["ts"].unique().tolist()
        self.logger.info("Found %d unique time step", len(ts))

        # predict per time step
        pred_series = []

        self.logger.info("Starting prediction...")
        for time in ts:
            # get data for time step
            df_step = df[df["ts"] == time].copy().set_index("zone_id")

            # derive features
            X = loader.derive(df_step)
            X = X.drop(columns=["zone_id", "ts", "target"], errors="ignore")
            X = loader.transform(df_step)

            # run predictions
            y_pred = model.predict(X)

            # create series
            pred_series.append(pd.Series(y_pred, name=time.strftime("%Y-%m-%d"), index=df_step.index))

        # construct prediction dataframe
        df_pred = pd.concat(pred_series, axis=1)
        df_geom_pred = df_geom[["ZONE_ID", "geometry"]].copy()
        df_geom_pred = df_geom_pred.merge(df_pred, left_on="ZONE_ID", right_index=True, how="left") \
            .fillna(0)
        
        print(df_geom_pred)

        # save to shapefile
        self.logger.info("Saving polygon...")
        df_geom_pred.to_file(self.config.output_file, driver="ESRI Shapefile")

        
