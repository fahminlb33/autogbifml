# AutoGBIFML

Welcome to AutoGBIFML project!

AutoGBIFML is a tool to create machine learning classification model to classify GBIF occurence data (DarwinCore format) with Copernicus Marine Data as the input data. This tool is part of my graduate thesis and currently still in work in progress phase.

Use the file tree on the left side to navigate this documentation.

https://github.com/pditommaso/awesome-pipeline

joblib - parallel engine
parsl - parallel workflow engine with graph execution


processing.run("native:zonalstatisticsfb", {'INPUT':'D:/gis/KAB. BOGOR/ADMINISTRASIDESA_AR_25K.shp','INPUT_RASTER':'D:/gis/temperature.tif','RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[0,1,2],'OUTPUT':'TEMPORARY_OUTPUT'})

qgis_process run native:zonalstatisticsfb --distance_units=meters --area_units=m2 --ellipsoid=EPSG:7030 --INPUT='D:/gis/KAB. BOGOR/ADMINISTRASIDESA_AR_25K.shp' --INPUT_RASTER='D:/gis/temperature.tif' --RASTER_BAND=1 --COLUMN_PREFIX=_ --STATISTICS=0 --STATISTICS=1 --STATISTICS=2 --OUTPUT=TEMPORARY_OUTPUT

processing.run("grass7:r.stats.zonal", {'base':'D:/gis/temperature.tif','cover':'D:/gis/temperature.tif','method':0,'-c':False,'-r':False,'output':'TEMPORARY_OUTPUT','GRASS_REGION_PARAMETER':None,'GRASS_REGION_CELLSIZE_PARAMETER':0,'GRASS_RASTER_FORMAT_OPT':'','GRASS_RASTER_FORMAT_META':''})

## Commands

autogbifml

gbifmeta - input Darwin, return bounding box, time extent, spatial extent, dll

download - downloads data from copernicus

prepare - input darwin, netcdf, and zone polygon, outputs well prepared parquet for training

eval - perform cross validation

train - train model

predct - input two raster and zone polygon? predict on single time or series

## Dataset

World administrative boundaries https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/

## Membuat poligon zona

1. Konversi format DarwinCore ke lat lon CSV
2. Project baru QGIS dengan CRS EPSG:4326 (WG 84)
3. Import kenampakan lat lon ke QGIS
4. Import XYZ Layer > OpenStreetMap
5. Import world administrative boundaries Shapefile. output=world_adm
6. Membuat bounding box dari koordinat kenampakan. Processing > Layer tools > Extract layer extent. output=bbox
7. Buffer bbox. Processing > Vector geometry > Buffer. input=bbox output=bbox_buffer distance=2 deg endcap=square join=mitter mitterlimit=2
8. Membuat zona darat yang akan dihapus. Processing > Vector overlay > Intersect. input=bbox_buffer overlay=world_adm output=bbox_remove_int
9. Membuat zona laut. Processing > Vector overlay > Difference. input=bbox_buffer overlay=bbox_intersect output=bbox_poly
10. Membuat grid. Processing > Vector creation > Create grid. type=rect/hex extent=bbox_poly horizontalSpacing=0,3 verticalSpacing=0,3 output=grid
11. Membuat memotong grid sesuai bounding box. Processing > Vector overlay > Intersect. input=grid overlay=bbox_poly output=grid_poly
12. Membuat poligon untuk memotong tengah laut. New Shapefile and then draw manually, output=bbox_remove_center
13. Memotong poligon grid dengan tengah laut. Processing > Vector overlay > Difference. input=grid_poly overlay=bbox_remove_center output=grid_poly_segments

grid_poly_segments adalah poligon zona untuk membuat data training dan testing dengan zonal statistics dan raster

--------

1. Layer Tools > Extract Layer Extent
2. Vector geometry > Bufffer [distance=1 deg, join style=mitter]
3. Vector overlay > Intersection [world and bb -> land area]
4. Research tool > Create grid [v/h spacing=0,83]
5. Vector overlay > Symmetrical difference [input=land, overlay=grid -> sea grid]
6. Vector analysis > Count points in polygon

https://carpentries-incubator.github.io/geospatial-python/10-zonal-statistics.html
