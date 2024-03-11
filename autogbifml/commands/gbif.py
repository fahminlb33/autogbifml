import re
from zipfile import ZipFile

import numpy as np
import pandas as pd


class GBIFMetaCommand:

    def __call__(self, file_path: str, output_filename: str) -> None:
        # read DarwinCore zip file
        with ZipFile(file_path, "r") as zip_ref:
            # read citations file
            with zip_ref.open("citations.txt") as citations_file:
                citations = citations_file.read().decode("utf-8")
                print(citations)

            # read occurence file
            with zip_ref.open("occurrence.txt") as occurrence_file:
                # read occurrence file
                df = pd.read_csv(occurrence_file, sep="\t")
                df["ts"] = pd.to_datetime(df["eventDate"], format='ISO8601')

                # print min max ts
                print(f"Min ts: {df['ts'].min()}")
                print(f"Max ts: {df['ts'].max()}")

                # if decimalLatitude and decimalLongitude are null, try to use footprintWKT
                df["latitude"] = df.apply(
                    GBIFMetaCommand.coalesce_coordinate("decimalLatitude"),
                    axis=1)
                df["longitude"] = df.apply(
                    GBIFMetaCommand.coalesce_coordinate("decimalLongitude"),
                    axis=1)

                # subset dataset
                df_subset = df[[
                    "occurrenceID", "ts", "latitude", "longitude", "species"
                ]]

                # sort by ts descending
                df_subset = df_subset.sort_values("ts", ascending=False)

                # save to csv
                df_subset.to_csv(output_filename, index=False)
                print(f"File saved to {output_filename}")

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
