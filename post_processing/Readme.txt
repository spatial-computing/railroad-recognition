steps to rasterize vector data and generate test data
1) load the orignal geo image
2) create a black raster image using qgis(menu-raster-raster calculator-> select any band from geo_image and enter exp="")
3) using the shapefile to add vector points to the above raster:  
C:\Users\Vinil\Desktop\vinil\temp>gdal_rasterize -b 1 -burn 255 -l rail rail.shp new_raster.tif
4) convert to jpg
gdal_translate -of JPEG C:\Users\Vinil\Desktop\vinil\temp\new_raster.tif C:/Users/Vinil/Desktop/vinil/temp/test.jpg
5) run the python script to generate the center co-ordinates of the pos and neg samples. 
The code to crop the image using image- and geo-coodinates is also present in script.
