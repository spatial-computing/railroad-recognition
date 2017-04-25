
from GenerateTrainingData import  Training_Data_generator


#sample usage
#to generate the training use the following function_calls
#change the necessary path in the input_data.txt which acts like a config file
#it will generate the following files in the output directory
# 1) bounding coordinates
# 2) positive and negative coordinates
# 3) rasterised data in png and tif format
# in the input_data, for the shapefiles give the path of the directory containing shapefiles of a particular type,
# all of them will be rasterised with the same value
# if you are using a shapefile created manually, just make sure its name start with "manual" eg manual_roads_vector.shp
# _
training_data= Training_Data_generator("input_data.txt")
training_data.generate_bounding_coordinates()
training_data.create_blank_raster()
training_data.rasterize_data()
training_data.generate_positive_negative_coordinates()

# you can use the Training_Data_generator to rasterise a sinle vector as well
# You can just pass the vector_path, burn value , buffer_size(optional, default is 0) and output image path(optional) as shown below(
# if you want to rasterise multiple vectors, pass the output_image_path param only in the last call

# training_data= Training_Data_generator("input_data.txt")
# path=r"C:\Users\Vinil\Desktop\vinil\data\roads_new\Trans_RoadSegment9_clip_proj.shp"
# path2=r"C:\Users\Vinil\Desktop\vinil\data\roads_new\Trans_RoadSegment10_clip_proj.shp"
# training_data.rasterize_single_vector(path,255,10)
# training_data.rasterize_single_vector(path2,235,10,"output/temp.png")