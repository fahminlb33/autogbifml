import logging

import yaml
import copernicusmarine

from joblib import Parallel, delayed
from pydantic import BaseModel

from infrastructure.logger import init_logger


class DownloadDataset(BaseModel):
    id: str
    variables: list[str]


class DownloadConfigRoot(BaseModel):
    minimumLongitude: float
    maximumLongitude: float
    minimumLatitude: float
    maximumLatitude: float
    minimumDepth: float
    maximumDepth: float
    startDateTime: str
    endDateTime: str
    outputPath: str
    dataset: list[DownloadDataset]


class DownloadCommandOptions(BaseModel):
    config_file: str
    jobs: int = 1


class DownloadCommand:

    def __init__(self, **kwargs) -> None:
        self.logger = init_logger("DownloadCommand")

    def __call__(self, **kwargs) -> None:
        self.config = DownloadCommandOptions(**kwargs)

        # read config
        dl_config = DownloadConfigRoot(
            **yaml.safe_load(open(self.config.config_file, "r")))

        # download data
        job_fn = delayed(DownloadCommand.download_data)
        jobs = [job_fn(d, self.config) for d in dl_config.dataset]
        Parallel(n_jobs=self.config.jobs, verbose=1)(jobs)

        self.logger.info("Download complete!")

    @staticmethod
    def disable_copernicus_log(self) -> None:
        logging.getLogger("copernicus_marine_blank_logger").disabled = True
        logging.getLogger("copernicus_marine_root_logger").disabled = True

    @staticmethod
    def download_data(dataset: DownloadDataset,
                      config: DownloadConfigRoot) -> None:
        # setup copernicus config
        DownloadCommand.disable_copernicus_log()

        # start subset download
        print(f"Downloading {dataset.id}...")
        out_path = copernicusmarine.subset(
            # dataset
            dataset_id=dataset.id,
            variables=dataset.variables,

            # slicer
            start_datetime=config.startDateTime,
            end_datetime=config.endDateTime,
            minimum_longitude=config.minimumLongitude,
            maximum_longitude=config.maximumLongitude,
            minimum_latitude=config.minimumLatitude,
            maximum_latitude=config.maximumLatitude,
            minimum_depth=config.minimumDepth,
            maximum_depth=config.maximumDepth,
            output_directory=config.outputPath,
            force_download=True,
            overwrite_output_data=False,
            # disable_progress_bar=True,
        )

        print(f"Saved {dataset.id} to {out_path}")
