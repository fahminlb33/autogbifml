import click

from commands import (DownloadCommand, PreprocessCommand, EvalCommand, GBIFMetaCommand, PredictCommand, TrainCommand)

@click.group()
def cli():
    pass


@cli.command()
@click.argument("file_path", type=str)
@click.argument("output_filename", type=str)
def gbifmeta(*args, **kwargs):
    GBIFMetaCommand()(*args, **kwargs)


@cli.command()
@click.option("--dataset-id", type=str)
@click.option("--output-path", type=str)
@click.option("--dt-start", default=None, help="Start datetime", type=str)
@click.option("--dt-end", default=None, help="End datetime", type=str)
@click.option("--min-lon", default=None, help="Minimum longitude", type=float)
@click.option("--max-lon", default=None, help="Maximum longitude", type=float)
@click.option("--min-lat", default=None, help="Minimum latitude",  type=float)
@click.option("--max-lat", default=None, help="Maximum latitude", type=float)
@click.option("--min-depth", default=None, help="Minimum depth", type=float)
@click.option("--max-depth", default=None, help="Maximum depth", type=float)
@click.option("--variables", default=[], multiple=True, help="Variables to download")
@click.option("--jobs", default=1, help="Number of jobs to run in parallel", type=int)
def download(*args, **kwargs):
    DownloadCommand(**kwargs)()


@cli.command()
@click.option("--raster-path", type=str, help="Raster path containing netCDF files")
@click.option("--vector-path", type=str, help="Vector path containing shapefile")
@click.option("--occurence-path", type=str, help="Occurence path containing csv files")
@click.option("--output-path", type=str, help="Output path")
@click.option("--jobs", type=int, help="Number of jobs to run in parallel", default=1)
def preprocess(*args, **kwargs):
    PreprocessCommand(**kwargs)()


if __name__ == "__main__":
    cli()
