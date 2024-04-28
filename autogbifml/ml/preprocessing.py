import os
from enum import Enum
from typing import Any

import xarray as xr
import pandas as pd
import geopandas as gpd
import xrspatial as xrs
import datashader as ds
import dask.dataframe as dd

from pydantic import BaseModel

from infrastructure.logger import init_logger


class DownsampleEnum(str, Enum):
    NONE = "none"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ZonalProcessorOptions(BaseModel):
    raster_file: str
    occurence_file: str
    zone_file: str
    output_path: str

    downsample: DownsampleEnum = DownsampleEnum.NONE


class ZonalProcessor:
    config: ZonalProcessorOptions

    ds_raster: xr.Dataset
    df_zone: gpd.GeoDataFrame
    df_occurence: gpd.GeoDataFrame

    zonal_mask: xr.DataArray
    occurence_dates: list[str]

    raster_has_depth: bool
    raster_variables: list[str]
    raster_dates: list[str]
    raster_depth: float

    def __init__(self, config: ZonalProcessorOptions) -> None:
        self.config = config
        self.logger = init_logger("ZonalProcessor")

    def __call__(self) -> Any:
        self.logger.info("Calculating zonal stats for %s",
                         os.path.basename(self.config.raster_file))

        # open dataset
        self.load_occurence()
        self.load_raster()
        self.load_zonal_mask()

        # process each day and variable
        for variable in self.raster_variables:
            self.logger.info("Processing variable: %s", variable)

            # to store results
            dfs = []

            # process all days
            skipped_dates = []
            for date in self.occurence_dates:
                # check if date is in dataset
                if date not in self.raster_dates:
                    skipped_dates.append(date)
                    continue

                # select data for variable, date, and possibly depth
                if not self.raster_has_depth:
                    zs = self.ds_raster[variable].sel(time=date)
                else:
                    zs = self.ds_raster[variable].sel(time=date,
                                                      depth=self.raster_depth)

                # calculate zonal statistics
                zs = xrs.zonal_stats(self.zonal_mask, zs, zone_ids=self.df_zone["ZONE_ID"]) \
                    .dropna() \
                    .drop(columns=["count"]) \
                    .add_prefix(f"{variable}_") \
                    .rename(columns={f"{variable}_zone": "zone_id"}) \
                    .assign(ts=date)

                # count number of occurence in zone
                zso: gpd.GeoDataFrame = gpd.sjoin(self.df_zone,
                                                  self.df_occurence,
                                                  how="left")
                zso = zso.groupby("ZONE_ID")["ts"] \
                    .count() \
                    .reset_index() \
                    .rename(columns={"ts": "target", "ZONE_ID": "zone_id"})

                # merge with zs
                zsm = zs.merge(zso, on="zone_id", how="left")
                zsm["target"] = (zsm["target"] > 0).astype(int)

                # append
                dfs.append(zsm)

            # merge results
            df_stats = pd.concat(dfs, ignore_index=True) \
                .reset_index() \
                .drop(columns=["level_0", "index"], errors="ignore")

            # parse date
            df_stats["ts"] = pd.to_datetime(df_stats["ts"])

            # save to parquet
            df_stats.to_parquet(
                f"{self.config.output_path}/zs_{variable}.parquet",
                engine="pyarrow")

            self.logger.warning("There are %d skipped dates",
                                len(skipped_dates))
            self.logger.info("Done processing variable: %s", variable)

    def load_occurence(self):
        self.logger.info(f"Loading occurence dataset...")

        # load occurence dataset
        df = pd.read_csv(self.config.occurence_file,
                         parse_dates=["ts"]).drop(columns=["species"])

        # resample if needed
        if self.config.downsample == DownsampleEnum.WEEKLY:
            df = df.resample("W", on="ts").mean().reset_index()
        elif self.config.downsample == DownsampleEnum.MONTHLY:
            df = df.resample("ME", on="ts").mean().reset_index()

        # get unique dates
        self.occurence_dates = df["ts"].dt.strftime(
            "%Y-%m-%d").sort_values().unique().tolist()

        # create points
        self.df_occurence = gpd.GeoDataFrame(df,
                                             geometry=gpd.points_from_xy(
                                                 df["longitude"],
                                                 df["latitude"]),
                                             crs="epsg:4326")

        self.logger.info("Found %d unique occurence dates",
                         len(self.occurence_dates))

    def load_raster(self):
        self.logger.info(f"Loading raster dataset...")

        # load raster dataset
        dsr = xr.open_dataset(self.config.raster_file)

        # resample if needed
        if self.config.downsample == DownsampleEnum.WEEKLY:
            dsr = dsr.resample(time="W").mean(dim="time")
        elif self.config.downsample == DownsampleEnum.MONTHLY:
            dsr = dsr.resample(time="ME").mean(dim="time")

        self.ds_raster = dsr

        # get variables
        self.raster_variables = list(dsr.keys())
        self.raster_dates = dsr["time"].dt.strftime("%Y-%m-%d").values.tolist()

        self.raster_has_depth = False
        if len(dsr[self.raster_variables[0]].shape) == 4:
            self.logger.info(f"This raster has depth coordinate!")
            self.raster_has_depth = True
            self.raster_depth = dsr[self.raster_variables[0]]["depth"][0]

    def load_zonal_mask(self):
        self.logger.info(f"Loading zonal dataset...")

        # load zone id polygon
        self.df_zone = gpd.read_file(self.config.zone_file)

        # check width, height
        w = self.ds_raster[self.raster_variables[0]].shape[1]
        h = self.ds_raster[self.raster_variables[0]].shape[2]
        if self.raster_has_depth:
            w = self.ds_raster[self.raster_variables[0]].shape[2]
            h = self.ds_raster[self.raster_variables[0]].shape[3]

        # create canvas as mask
        canvas = ds.Canvas(
            plot_width=w,
            plot_height=h,
            y_range=(float(self.ds_raster.coords["latitude"].min().values),
                     float(self.ds_raster.coords["latitude"].max().values)),
            x_range=(float(self.ds_raster.coords["longitude"].min().values),
                     float(self.ds_raster.coords["longitude"].max().values)))

        # create polygon mask using geometry from shapefile
        self.zonal_mask = canvas.polygons(self.df_zone,
                                          geometry="geometry",
                                          agg=ds.max("ZONE_ID"))
