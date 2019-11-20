The training data generation component takes maps and auxiliry data (e.g., contemperary vector data) to automatically generate labeled data
for object recognition. However, the auxiliary data usually only provide approximate or weak annotations. For the polyline objects, 
the vector data usually align to the surrounding places of the target object on the map because of the misalignment problem. 
For the polygon objects, the vector data usualy provide a region-of-interest covering lots of target objects.
Therefore, the task of this component is to solve misalignment problem for polyline objects,
and generate precise bounding boxes for polygon objects from region-level annotations.

**alignment.py** is for polyline object.

**Implementation instruction** (Note all parameters should be set at the begining of the scripts)

**First**, run **alignment_data_generator.py** 

  Inputs: georeferenced map, polyline vector data
  
  Outputs: the map in .png format, vector data in .txt format storing vector points in pixel coordinate

**Second**, run **alignment.py**
  
  Inputs: map in .png format, the directory stoering vector points
  
  Outputs: a folder for aligned vector points
  
**Third**, run **alignment_rasterization.py**

  Inputs: map in .png format, the output folder
  
  Outputs: annotation for each pixel 
  
 
**Generate both positive (target) and negativev (non-target) samples for recognition model**

**training_data_generator.py**

Two functions:

**generator_from_txt_folder**: take the folder to generate a txt file for positive samples

**generator_from_shp**: take georeferenced map and vector data of other objcts to generate a txt file for negative samples

**Dependencies:**
opencv
osgeo
gdal
gdal_const
