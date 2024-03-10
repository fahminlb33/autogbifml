import logging

import pandas as pd
import copernicusmarine

from pydantic import BaseModel
from joblib import Parallel, delayed

class DownloadCommandOptions(BaseModel):
   dataset_id: str
   output_path: str

   variables: list[str]

   dt_start: str
   dt_end: str

   min_lon: float
   max_lon: float
   min_lat: float
   max_lat: float
   min_depth: float | None = None
   max_depth: float | None = None

   jobs: int = 1

class DownloadCommand:
   def __init__(self, **kwargs) -> None:
      self.config = DownloadCommandOptions(**kwargs)

   def __call__(self) -> None:
      # prepare copernicus config
      self.prepare()

      # create date range
      date_range = pd.interval_range(start=pd.to_datetime(self.config.dt_start), end=pd.to_datetime(self.config.dt_end), freq="ME").tolist()
      date_range = [(date.left.strftime('%Y-%m-%dT%H:%M:%S'), date.right.strftime('%Y-%m-%dT%H:%M:%S')) for date in date_range]

      # download data
      Parallel(n_jobs=self.config.jobs, verbose=1)(delayed(DownloadCommand.download_data)(self.config, date) for date in date_range)

   def prepare(self) -> None:
      logging.getLogger("copernicus_marine_blank_logger").disabled = True
      logging.getLogger("copernicus_marine_root_logger").disabled = True

   @staticmethod
   def download_data(config: DownloadCommandOptions, dt: tuple[str, str]) -> None:
      filename = f"{config.dataset_id}-{dt[0][:7]}_{dt[1][:7]}.nc"
      print(f"Downloading {filename}")

      copernicusmarine.subset(
         dataset_id = config.dataset_id,
         variables = config.variables,
         start_datetime = dt[0],
         end_datetime = dt[1],
         minimum_longitude = config.min_lon,
         maximum_longitude = config.max_lon, 
         minimum_latitude = config.min_lat,
         maximum_latitude = config.max_lat, 
         output_filename = filename,
         output_directory = config.output_path,
         force_download=True,
         overwrite_output_data=True,
         # disable_progress_bar=True,
      )
