import os
from IPython import embed
from helper_methods import run_and_return_output
from download_raster_map import download_raster_map
from geo_reference_raster_map import geo_reference_raster_map
from prepare_vector_data import prepare_vector_data

download_url = "https://prd-tnm.s3.amazonaws.com/hm_archive/CA/CA_Bray_100414_2001_24000_bag.zip"
quadrangle_state = "California"
quadrangle_area = "Bray"

working_dir = download_raster_map(download_url)
map_name = os.path.split(working_dir)[-1].replace("_bag", "")
print working_dir
print map_name


# Geo-reference raster map
original_tif = os.path.join(working_dir, "data", map_name + "_orig.tif")
gcp_xml = os.path.join(working_dir, "data", map_name + "_gcp.xml")

if not os.path.exists(original_tif):
    raise RuntimeError("Expected to find original map in " + original_tif + " but it wasn't there")

if not os.path.exists(gcp_xml):
    raise RuntimeError("Expected to find gcp xml in " + gcp_xml + " but it wasn't there")


print "Geo-referencing raster map " + original_tif
geo_reference_raster_map(original_tif, gcp_xml)

print "Preparing vectorized data"
prepare_vector_data(working_dir, quadrangle_state, quadrangle_area)