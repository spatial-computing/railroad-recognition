# Instructions for wetlands detection in USGS topographic map archive
## The goal is to extract the locations of wetland symbols on the maps and convert the extraction results into vector data stored as shapefile


### 1) Rerequired libraries
Keras == 2.1.6 \\
tensorflow-gpu == 1.2.0 \\
cuda == 8.0 \\
osgeo python library

### 2) Run scripts
### run detection.py: the inputs are the map and trained model path, the output is a rasterized detection image which has the the same size as the input map
python detection.py --test_map_path path_to_the_map --model_path trained_model_path --pred_path path_to_save_the_detection_res

The trained model can be downloaded from https://www.dropbox.com/s/h7o01if5ty0cc6i/wetlands_orcutt.hdf5?dl=0

### run polygonization.py: convert the detection reesults into the shapefile which projection system is the same as the input map
python polygonization.py --pred_path path_to_the_detection_res --tif_map_path path_to_the_tif_map

### 3) Download data
The link to download maps (USGS topographic map archive): https://ngmdb.usgs.gov/topoview/viewer/#4/40.00/-100.00

