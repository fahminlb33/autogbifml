import os
from enum import Enum
from typing import Any

import xarray as xr
import pandas as pd
import geopandas as gpd
import xrspatial as xrs
import datashader as ds

from pydantic import BaseModel


class ZonalRasterProcessor:
    type: str
    derived_vars: list[str] = ["mean", "sum", "count"]
    selected_vars: list[str]

    raster_file: str
    occurrence_file: str
    zone_file: str
    output_path: str

    ds_raster: xr.Dataset
    df_zone: gpd.GeoDataFrame
    df_occurence: gpd.GeoDataFrame

    zonal_mask: xr.DataArray
    occurence_dates: list[str]

    raster_has_depth: bool
    raster_dates: list[str]
    raster_depth: float

    def __init__(
        self,
        type: str,
        derived_vars: list[str],
        selected_vars: list[str],
        raster_file: str,
        occurrence_file: str,
        zone_file: str,
        output_path: str,
    ) -> None:
        self.pid = f"ZonalProcessor-{os.getpid()}"

        self.type: str = type
        self.derived_vars: list[str] = derived_vars
        self.selected_vars: list[str] = selected_vars

        self.raster_file = raster_file
        self.occurrence_file = occurrence_file
        self.zone_file = zone_file
        self.output_path = output_path

    def __call__(self) -> Any:
        print(
            f"{self.pid}: Calculating zonal stats for {os.path.basename(self.raster_file)}"
        )

        # open dataset
        self.load_occurence()
        self.load_raster()
        self.load_zonal_mask()
        self.create_zone_target()

        if self.type == "cmems":
            self.derive_cmems()
        else:
            self.derive_gebco()

    def derive_gebco(self):
        # process variable
        for variable in self.selected_vars:
            new_variable = (
                os.path.basename(self.raster_file).split(".")[0].split("-")[1]
            )
            print(f"{self.pid}: Processing variable: {new_variable}")

            # calculate zonal statistics
            zs = (
                xrs.zonal_stats(
                    self.zonal_mask,
                    self.ds_raster[variable],
                    zone_ids=self.df_zone["ZONE_ID"],
                    stats_funcs=self.derived_vars,
                )
                .dropna()
                .drop(columns=["count"])
                .add_prefix(f"{variable}_")
                .rename(columns={f"{variable}_zone": "zone_id"})
            )

            # merge with zs
            zsm = zs.merge(self.zone_target, on="zone_id", how="left")
            zsm["target"] = (zsm["target"] > 0).astype(int)

            # rename cols
            zsm.columns = [x.replace(variable, new_variable) for x in zsm.columns]

            # save to parquet
            zsm.to_parquet(f"{self.output_path}/gebco_{new_variable}.parquet")

    def derive_cmems(self):
        # process each day and variable
        for variable in self.selected_vars:
            print(f"{self.pid}: Processing variable: {variable}")

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
                    rs = self.ds_raster[variable].sel(time=date)
                else:
                    rs = self.ds_raster[variable].sel(
                        time=date, depth=self.raster_depth
                    )

                # calculate zonal statistics
                zs = (
                    xrs.zonal_stats(
                        self.zonal_mask,
                        rs,
                        zone_ids=self.df_zone["ZONE_ID"],
                        stats_funcs=self.derived_vars,
                    )
                    .dropna()
                    .drop(columns=["count"])
                    .add_prefix(f"{variable}_")
                    .rename(columns={f"{variable}_zone": "zone_id"})
                    .assign(ts=date)
                )

                # merge with zs
                zsm = zs.merge(self.zone_target, on="zone_id", how="left")
                zsm["target"] = (zsm["target"] > 0).astype(int)

                # append
                dfs.append(zsm)

            # merge results
            df_stats = (
                pd.concat(dfs, ignore_index=True)
                .reset_index()
                .drop(columns=["level_0", "index"], errors="ignore")
            )

            # parse date
            df_stats["ts"] = pd.to_datetime(df_stats["ts"])

            # save to parquet
            df_stats.to_parquet(f"{self.output_path}/cmems_{variable}.parquet")

            print(f"{self.pid}: There are {len(skipped_dates)} skipped dates")
            print(f"{self.pid}: Done processing variable: {variable}")

    def load_occurence(self):
        print(f"{self.pid}: Loading occurence dataset...")

        # load occurence dataset
        df = pd.read_csv(self.occurrence_file, parse_dates=["ts"]).drop(
            columns=["species"]
        )

        # get unique dates
        self.occurence_dates = (
            df["ts"].dt.strftime("%Y-%m-%d").sort_values().unique().tolist()
        )

        # create points
        self.df_occurence = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="epsg:4326",
        )

        print(f"{self.pid}: Found {len(self.occurence_dates)} unique occurence dates")

    def load_raster(self):
        print(f"{self.pid}: Loading raster dataset...")

        # load raster dataset
        self.ds_raster = xr.open_dataset(self.raster_file)

        # validate variables
        if len(set(self.selected_vars).intersection(set(self.ds_raster.keys()))) < len(
            self.selected_vars
        ):
            missing_vars = set(self.ds_raster.keys()) - set(self.selected_vars)
            raise ValueError(
                f"Some selected variables not found in raster dataset! Missing: {missing_vars}"
            )

        # special treatment for CMEMS data
        if self.type == "cmems":
            # get occurrence dates
            self.raster_dates = (
                self.ds_raster["time"].dt.strftime("%Y-%m-%d").values.tolist()
            )

            # check if raster is 3D
            self.raster_has_depth = False
            if len(self.ds_raster[self.selected_vars[0]].shape) == 4:
                print(f"{self.pid}: This raster has depth coordinate!")
                self.raster_has_depth = True
                self.raster_depth = self.ds_raster[self.selected_vars[0]]["depth"][0]

    def load_zonal_mask(self):
        print(f"{self.pid}: Loading zonal mask...")

        latitude_coord = "latitude" if self.type == "cmems" else "lat"
        longitude_coord = "longitude" if self.type == "cmems" else "lon"

        # load zone id polygon
        self.df_zone = gpd.read_file(self.zone_file)

        # check width, height
        h = self.ds_raster.coords[latitude_coord].shape[0]
        w = self.ds_raster.coords[longitude_coord].shape[0]

        # create canvas as mask
        canvas = ds.Canvas(
            plot_width=w,
            plot_height=h,
            y_range=(
                float(self.ds_raster.coords[latitude_coord].min().values),
                float(self.ds_raster.coords[latitude_coord].max().values),
            ),
            x_range=(
                float(self.ds_raster.coords[longitude_coord].min().values),
                float(self.ds_raster.coords[longitude_coord].max().values),
            ),
        )

        # create polygon mask using geometry from shapefile
        self.zonal_mask = canvas.polygons(
            self.df_zone, geometry="geometry", agg=ds.max("ZONE_ID")
        )

    def create_zone_target(self):
        # count number of occurence in zone
        zso: gpd.GeoDataFrame = gpd.sjoin(self.df_zone, self.df_occurence, how="left")

        self.zone_target = (
            zso.groupby("ZONE_ID")["ts"]
            .count()
            .reset_index()
            .rename(columns={"ts": "target", "ZONE_ID": "zone_id"})
        )
