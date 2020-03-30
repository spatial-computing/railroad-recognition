**The pipeline is for cotent extraction from georeferenced documents.**

The pipeline has three components: training data generation, content extraction, content vecterization

**Training data gneration:**

  Inputs: georeferenced maps, auxiliary data for annotation
  
  Outputs: annotated positive and negative data for content extraction

**Semantic Segmentation:**

  Inputs: maps, annotations, pixel coordinates for both positive and negeative samples
  
  Outputs: The testing result for the map
  
  Note: for semantic segmentation, the annotation is a black-white image the same size as the map
        the output is also a black-white image, whites are predicted positive.
        
**Vectorization:**

  Inputs: georeferenced maps, recognition results
  
  Outputs: the shapefile, the same coordinate system as georeferenced maps
  
The detail comments (dependent libraries, scripts instruction) for each component is in the corresponding folder. 
