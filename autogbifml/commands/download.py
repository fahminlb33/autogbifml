from typing import Any
from argparse import ArgumentParser, _SubParsersAction

import yaml
import copernicusmarine
from pydantic import BaseModel

from services.base import BaseCommand


class DownloadCmemsCommandOptions(BaseModel):
    profile_file: str


class DownloadCmemsCommand(BaseCommand):
    def __init__(self) -> None:
        super(DownloadCmemsCommand, self).__init__()

    @staticmethod
    def add_parser(subparser: _SubParsersAction):
        parser: ArgumentParser = subparser.add_parser(
            "download", help="Download CMEMS dataset"
        )

        parser.set_defaults(func=DownloadCmemsCommand())
        parser.add_argument("profile_file", type=str, help="Path to download profile")

    def __call__(self, args) -> Any:
        # parse args
        self.config = DownloadCmemsCommandOptions(**vars(args))

        # load profile
        self.logger.info("Loading download profile...")
        profile = yaml.load(open(args.profile_file, "r"))

        # process all profiles
        for dataset in profile["dataset"]:
            self.logger.info(f"Download start: {dataset.id}")
            copernicusmarine.subset(
                dataset_id=dataset.id,
                variables=dataset.variables,
                start_datetime=f"{profile.startDateTime}T00:00:00",
                end_datetime=f"{profile.endDateTime}T23:59:59",
                minimum_longitude=profile.minimumLongitude,
                maximum_longitude=profile.maximumLongitude,
                minimum_latitude=profile.minimumLatitude,
                maximum_latitude=profile.maximumLatitude,
                minimum_depth=profile.minimumDepth,
                maximum_depth=profile.maximumDepth,
            )
