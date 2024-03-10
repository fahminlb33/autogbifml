import glob
import json
from typing import Any

import rasterio
import pandas as pd
import xarray as xr
import geopandas as gpd

from joblib import Parallel, delayed
from pydantic import BaseModel
from rasterstats import zonal_stats

class PreprocessCommandOptions(BaseModel):
   raster_path: str
   vector_path: str
   occurence_path: str
   output_path: str

   jobs: int = 1


class PreprocessCommand:
    ZONAL_STATS = ["min", "max", "mean", "median", "std", "range"]

    def __init__(self, **kwargs) -> None:
        self.config = PreprocessCommandOptions(**kwargs)
    
    def __call__(self) -> None:
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
            for date in self.occurence_dates:                
                # check if date is in dataset
                if date not in ds.time:
                    continue

                # convert to string
                date_str = date.strftime("%Y-%m-%d")

                # create delayed function
                delayed_func = delayed(PreprocessCommand.zonal_stats)
                jobs = [
                    delayed_func(
                        f"{variable}_",
                        self.zonal_index, 
                        self.zonal_mask, 
                        ds.sel(time=date)[variable].values, 
                        self.crs_affine, 
                        self.ZONAL_STATS, 
                        f"{self.config.output_path}/{variable}_{date_str}.csv"
                    ) for variable in list(ds.keys())
                ]

                # run in parallel
                Parallel(n_jobs=self.config.jobs, verbose=1)(jobs)

    def load_vector_data(self) -> None:
        # get grid mask
        grid_df = gpd.read_file(self.config.vector_path)
        self.zonal_mask = grid_df["geometry"]
        self.zonal_index = grid_df["id"].astype(int)

        # derived from EPSG:4326
        self.crs_affine = rasterio.Affine(0.25, 0.0, 18.875, 0.0, -0.25, -0.875)

    def load_occurence_data(self) -> None:
        # load occurence data
        occurence_df = pd.read_csv(self.config.occurence_path, parse_dates=["ts"])

        # get all unique occurence date
        occurence_df_subset = occurence_df[(occurence_df["ts"].dt.year >= 2021) & (occurence_df["ts"].dt.month >= 9)]
        self.occurence_dates = occurence_df_subset["ts"].unique()

        print(f"Found {len(self.occurence_dates)} unique occurence dates")
    
    @staticmethod
    def zonal_stats(prefix, zonal_index, zonal_mask, zonal_values, affine, stats, output_path):
        print(f"Calculating zonal stats for {output_path}")

        # calculate zonal index
        result = zonal_stats(zonal_mask, zonal_values, affine=affine, stats=stats)

        # combine with zonal index
        result_pd = pd.DataFrame(result).add_prefix(prefix)
        result_pd.index = zonal_index

        # drop na
        result_pd = result_pd.dropna()

        # save to file
        result_pd.to_csv(output_path)
