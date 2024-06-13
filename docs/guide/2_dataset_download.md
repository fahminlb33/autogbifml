# Dataset Information

## GBIF OBIS-SEAMAP Humpback Whale

### Spatial & Temporal Extent

Zona dibagi menjadi dua dari bbox_remove_center. Kedua zona memiliki grid yang sama.

Zone 1

Minimum longitude 19.150196563718747
Maximum longitude 59.982290053084995
Minimum latitude -40.060309775960881
Maximum latitude -1.374518636676697
Minimum-depth 0
Maximum-depth 1

Zone 2

Minimum longitude 104.00766264867093
Maximum longitude 144.211570084354634
Minimum latitude -40.269705127188409
Maximum latitude -1.583913987904232
Minimum-depth 0
Maximum-depth 1

## Copernicus Marine Service

### Downloading using CLI

copernicusmarine subset --dataset-id cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m --variable so --start-datetime 2021-09-01T00:00:00 --end-datetime 2023-11-30T00:00:00 --minimum-longitude 19 --maximum-longitude 60 --minimum-latitude -40 --maximum-latitude -1 --minimum-depth 0 --maximum-depth 1 --output-filename so.nc

python autogbifml download \
    --dataset-id cmems_mod_glo_phy_anfc_0.083deg_P1D-m \
    --output-path dataset/copernicus_marine \
    --variables={zos,pbo,tob,sob,siconc,sithick,sisnthick,ist,usi,vsi} \
    --dt-start 2021-09-01 \
    --dt-end 2023-11-30 \
    --min-lon 19 \
    --max-lon 60 \
    --min-lat -40 \
    --max-lat -1 \
    --jobs 4

### Product Catalog

#### Global Ocean Physics Analysis and Forecast

python autogbifml download \
    --dataset-id cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m \
    --output-path dataset/copernicus_marine \
    --variables={so} \
    --dt-start 2021-09-01 \
    --dt-end 2023-11-30 \
    --min-lon 19 \
    --max-lon 60 \
    --min-lat -40 \
    --max-lat -1 \
    --jobs 4

Temporal extent: 1 Nov 2020 to 19 Mar 2024
Downloadable: 2021-09-01 - 2023-11-30

- https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description
- https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf

cmems_mod_glo_phy_anfc_0.083deg_P1D-m
zos
pbo
tob
sob
siconc
sithick
sisnthick
ist
usi
vsi

cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m
so

cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m
thetao

cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m
uo
vo

cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m
wo

#### Global Ocean Biogeochemistry Analysis and Forecast

Temporal extent: 1 Oct 2021 to 15 Mar 2024
Downloadable: 

- https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_BGC_001_028/description
- https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-028.pdf

cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m
nppv
o2

cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m
talk
dissic
ph

cmems_mod_glo_bgc-co2_anfc_0.25deg_P1D-m
spco2

cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m
no3
po4
si
fe

cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m
chl
phyc

cmems_mod_glo_bgc-optics_anfc_0.25deg_P1D-m
kd

python autogbifml download \
    --dataset-id cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m \
    --output-path dataset/copernicus_marine \
    --variables=so \
    --dt-start 2021-09-01 \
    --dt-end 2023-11-30 \
    --min-lon 19 \
    --max-lon 60 \
    --min-lat -40 \
    --max-lat -1 \
    --jobs 4



## COMMAND DOWNLOAD

### Americas

copernicusmarine subset --dataset-id cmems_mod_glo_phy_anfc_0.083deg_P1D-m --variable={zos,pbo,tob,sob} --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m --variable so --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m --variable thetao --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m --variable={uo,vo} --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m --variable wo --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m --variable={nppv,o2} --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m --variable={talk,dissic,ph} --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_bgc-co2_anfc_0.25deg_P1D-m --variable spco2 --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m --variable={no3,po4,si,fe} --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30

copernicusmarine subset --dataset-id cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m --variable={chl,phyc} --minimum-longitude 20 --maximum-longitude 58 --minimum-latitude -34 --maximum-latitude 4 --minimum-depth 1 --maximum-depth 1 --start-datetime 2021-09-01 --end-datetime 2023-11-30
