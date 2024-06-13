# Preprocess

## Add ID to Zone

python autogbifml preprocess zone-id ./dataset/shp/america/sea_grid_count_americas.shp ./dataset/shp/america/sea_grid_count_americas_zoned.shp

## Calculate Zonal Statistics

python autogbifml --jobs 4 preprocess zonal-stats ./dataset/darwin_humpback_whale/occurence_proc.csv ./dataset/shp/america/sea_grid_count_americas_zoned.shp ./dataset/copernicus_marine/america ./dataset/zones_features/america_daily --temp-dir ./temp
