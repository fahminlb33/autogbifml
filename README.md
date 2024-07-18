# AutoGBIFML

> This repository contains the source code for the paper "Humpback Whale Occurrences Zone Classification based on Environmental Factors using Tree-based Algorithms"

AutoGBIFML is a library developed to train a machine learning classification model from GBIF occurrences data and Copernicus Marine Service (CMEMS) raster data.

Simplified workflow:

1. Download occurence data from GBIF
2. Create polygons that intersects the occurrences data (e.g. rectangular grid)
3. Derive a new zonal statistics dataset from GBIF data and the occurence polygons
4. Run a hyperparameter tuning on a machine learning algorithm (decision tree, random forest, CatBoost, XGBoost)
5. Run training using the best parameters from previous step
6. Run inference on new data

See [Docs](./docs/index.md) for more information.

## License

Licensed under Apache License Version 2.0.

## Citations

TBA.
