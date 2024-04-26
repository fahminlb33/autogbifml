import re
import glob
from zipfile import ZipFile

import rasterio
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import xrspatial as xrs

from joblib import Parallel, delayed
from pydantic import BaseModel

from infrastructure.logger import init_logger


class PreprocessGBIFCommandOptions(BaseModel):
    file_path: str
    output_filename: str


class PreprocessGBIFCommand:

    def __init__(self) -> None:
        self.logger = init_logger("PreprocessGBIFCommand")

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
                self.logger.info(
                    f"File saved to {self.config.output_filename}")

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


class PreprocessCopernicusCommandOptions(BaseModel):
    raster_path: str
    vector_path: str
    occurence_path: str
    output_path: str

    jobs: int = 1


class PreprocessCopernicusCommand:
    ZONAL_STATS = ["min", "max", "mean", "median", "std", "range"]

    def __init__(self, **kwargs) -> None:
        self.logger = init_logger("PreprocessCopernicusCommand")

    def __call__(self, **kwargs) -> None:
        self.config = PreprocessCopernicusCommandOptions(**kwargs)

        # load dataset
        self.load_vector_data()
        self.load_occurence_data()

        # find all raster files
        raster_files = glob.glob(f"{self.config.raster_path}/*.nc")
        print(f"Found {len(raster_files)} raster files")

        # process each raster file
        for raster_file in raster_files:
            # open raster file
            ds = xr.open_dataset(raster_file)

            # process each day and variable
            # TODO: allow time series resampling to monthly or weekly
            for date in self.occurence_dates:
                # check if date is in dataset
                if date not in ds.time:
                    continue

                # convert to string
                date_str = date.strftime("%Y-%m-%d")

                # create delayed function
                delayed_func = delayed(PreprocessCopernicusCommand.zonal_stats)
                jobs = [
                    delayed_func(
                        variable, self.zonal_index, self.zonal_mask,
                        ds.sel(time=date)[variable].values, self.crs_affine,
                        self.ZONAL_STATS,
                        f"{self.config.output_path}/{variable}_{date_str}.parquet"
                    ) for variable in list(ds.keys())
                ]

                # run in parallel
                Parallel(n_jobs=self.config.jobs, verbose=1)(jobs)

    def run_merge(self) -> None:
        pass

    def load_vector_data(self) -> None:
        # get grid mask
        grid_df = gpd.read_file(self.config.vector_path)
        self.zonal_mask = grid_df["geometry"]
        self.zonal_index = grid_df["id"].astype(int)

        # derived from EPSG:4326
        self.crs_affine = rasterio.Affine(0.25, 0.0, 18.875, 0.0, -0.25,
                                          -0.875)

    def load_occurence_data(self) -> None:
        # load occurence data
        occurence_df = pd.read_csv(self.config.occurence_path,
                                   parse_dates=["ts"])

        # get all unique occurence date
        # occurence_df = occurence_df[(occurence_df["ts"].dt.year >= 2021) & (occurence_df["ts"].dt.month >= 9)]
        self.occurence_dates = occurence_df["ts"].unique()

        print(f"Found {len(self.occurence_dates)} unique occurence dates")

    @staticmethod
    def zonal_stats(variable, zonal_index, zonal_mask, zonal_values, affine,
                    stats, output_path):
        print(f"Calculating zonal stats for {output_path}")

        # calculate zonal index
        result = zonal_stats(zonal_mask,
                             zonal_values,
                             affine=affine,
                             stats=stats)

        # combine with zonal index
        result_pd = pd.DataFrame(result) \
            .add_prefix(f"{variable}_") \
            .assign(variable=variable) \
            .set_index(zonal_index)

        # drop na
        result_pd = result_pd.dropna()

        # save to file
        result_pd.to_parquet(output_path)
