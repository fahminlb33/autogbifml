# AutoGBIFML

Welcome to AutoGBIFML project!

AutoGBIFML is a tool to create machine learning classification model to classify GBIF occurence data (DarwinCore format) with Copernicus Marine Data as the input data. This tool is part of my graduate thesis and currently still in work in progress phase.

See more about [dataset](./data.md) and the [delineation process](./whale_zone_delineation.md) used in this research.

## Installing AutoGBIFML

AutoGBIFML encourage the use of virtual environment. I personally use `uv` to create the venv.

```bash
git clone https://github.com/fahminlb33/autogbifml.git
cd autogbifml
pip install -r requirements.txt
sudo apt install gdal-bin
```

Converting GeoTIFF to NetCDF

```sh
gdal_translate -of netCDF -co 'FORMAT=NC4' ./dataset/africa/gebco/africa-aspect.tif ./dataset/africa/gebco/africa-aspect.nc
gdal_translate -of netCDF -co 'FORMAT=NC4' ./dataset/africa/gebco/africa-slope.tif ./dataset/africa/gebco/africa-slope.nc

gdal_translate -of netCDF -co 'FORMAT=NC4' ./dataset/australia/gebco/australia-aspect.tif ./dataset/australia/gebco/australia-aspect.nc
gdal_translate -of netCDF -co 'FORMAT=NC4' ./dataset/australia/gebco/australia-slope.tif ./dataset/australia/gebco/australia-slope.nc
```

## Commands

- `download` download data from Copernicus Marine Environmental Monitoring Service (CMEMS)
- `preprocess` input darwin, netcdf, and zone polygon, outputs well prepared parquet for training
- `tune` perform cross validation
- `train` train model
- `evaluate` evaluate a model
- `predict` input two raster and zone polygon? predict on single time or series

### `download`

Download dataset from Copernicus Marine Environmental Monitoring Service (CMEMS) using a download profile specification. The sample specs is available in the `profiles` directory.

Example command:

```sh
python autogbifml download profiles/download-africa.yml
python autogbifml download profiles/download-australia.yml
```

### `preprocess`

#### `preprocess occurrence`

Converts GBIF DarwinCore format into a simplified CSV occurrence data for training and testing the model.

Example command:

```sh
python autogbifml preprocess occurrence ./dataset/gbif/0013575-240202131308920.zip ./dataset/gbif/occurrence.csv
```

#### `preprocess zone-id`

Adds a unique ID to each polygon in the whale sighting zones. The input must be a Shapefile and contains one or more polygons.

Example command:

```sh
python autogbifml preprocess zone-id ./dataset/shp/africa/grid-sea-africa.shp ./dataset/shp/africa/grid-sea-africa-zoned.shp
python autogbifml preprocess zone-id ./dataset/shp/australia/grid-sea-australia.shp ./dataset/shp/australia/grid-sea-australia-zoned.shp
```

#### `preprocess zonal-stats`

Calculates the zonal statistics using the zoned whale sighting zone polygons and CMEMS raster data.

Example command:

```sh
python autogbifml --jobs 4 preprocess zonal-stats \
    ./dataset/gbif/occurrence.csv \
    ./dataset/shp/africa/grid-sea-africa-zoned.shp \
    ./dataset/africa/cmems \
    ./dataset/zonal/africa.parquet

python autogbifml --jobs 4 preprocess zonal-stats \
    ./dataset/gbif/occurrence.csv \
    ./dataset/shp/australia/grid-sea-australia-zoned.shp \
    ./dataset/australia/cmems \
    ./dataset/zonal/australia.parquet
```

#### `preprocess merge`

Merge different spatial location dataset from previous zonal statistics dataset. For example, it is used to combine the Africa and Australia dataset.

Example command:

```sh
python autogbifml preprocess merge ./dataset/zonal ./dataset/merged --test-size 0.2
```

#### `preprocess feature-selection`

Performs feature selection using information gain as the metric.

Example command:

```sh
python autogbifml preprocess feature-selection \
    ./dataset/merged/train.parquet \
    ./dataset/merged/train-sel-10.parquet \
    --strategy=topk \
    --topk=10

python autogbifml preprocess feature-selection \
    ./dataset/merged/test.parquet \
    ./dataset/merged/test-sel-10.parquet \
    --strategy=manual \
    --features=sob_mean,sob_sum,fe_mean,fe_sum,so_sum,po4_mean,pbo_mean,pbo_sum,tob_mean,tob_sum
```

## `tune`

Performs hyperparameter tuning using the training dataset.

Example command:

```sh
python autogbifml --jobs 12 tune \
    ./dataset/merged/train-sel-10.parquet \
    ./dataset/models/catboost_tune \
    --name thesis_catboost_tune_v3 \
    --algorithm catboost \
    --shuffle \
    --tracking-url http://10.20.20.102:8009 \
    --storage mysql://optuna:optuna123@10.20.20.102:3306/optuna
```

For a complete commands used in the reference paper, check the `scripts/tune.sh` script.

## `train`

Trains a model and evaluate it using train and test data. This will output the data loader, model, and feature importance CSV.
The input dataset MUST have two files with the name "train" and "test" with parquet extension.

Example command:

```sh
python autogbifml train \
    ./dataset/merged \
    ./dataset/models/rf \
    --algorithm random_forest \
    --params-file ./dataset/models/best_params_random_forest.yml
```

## `predict`

Performs an inference on input test data and outputs the predictions in GeoJSON format.

Example command:

```sh
python autogbifml predict \
    --algorithm random_forest \
    --loader-file ./dataset/models10/loader.joblib \
    --classifier-file ./dataset/models10/random_forest_model.joblib \
    --polygon-file ./dataset/africa/shp/grid-sea-africa-zoned.shp \
    --dataset-file ./dataset/sample-predict.parquet \
    --output-file ./predictions.json
```
