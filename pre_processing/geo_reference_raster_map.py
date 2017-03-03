import argparse
import xml.etree.ElementTree as ETree
import os
from helper_methods import run_and_return_output
from helper_methods import gdalsrsinfo
from helper_methods import gdal_translate
from helper_methods import gdalwarp
# How to run:
# python geo_reference_raster_map.py path/to/map.tif
# By default will create geo-referenced map in the same directory as the original map


def extract_gcp_coordinates(gcp_xml_location):
    tree = ETree.parse(gcp_xml_location)
    root = tree.getroot()

    gcp_args_dict = dict()
    mark_u = "MarkU"
    mark_v = "MarkV"
    mark_longitude = "MarkLongitude"
    mark_latitude = "MarkLatitude"

    for element in list(root):
        attribute_name = element.tag
        attribute_value = element.text

        if mark_u in attribute_name:
            argument_position = int(attribute_name.split(mark_u, 1)[1])
            if argument_position not in gcp_args_dict:
                gcp_args_dict[argument_position] = {}

            gcp_args_dict[argument_position][mark_u] = attribute_value

        if mark_v in attribute_name:
            argument_position = int(attribute_name.split(mark_v, 1)[1])
            if argument_position not in gcp_args_dict:
                gcp_args_dict[argument_position] = {}

            gcp_args_dict[argument_position][mark_v] = attribute_value

        if mark_longitude in attribute_name:
            argument_position = int(attribute_name.split(mark_longitude, 1)[1])
            if argument_position not in gcp_args_dict:
                gcp_args_dict[argument_position] = {}

            gcp_args_dict[argument_position][mark_longitude] = attribute_value

        if mark_latitude in attribute_name:
            argument_position = int(attribute_name.split(mark_latitude, 1)[1])
            if argument_position not in gcp_args_dict:
                gcp_args_dict[argument_position] = {}

            gcp_args_dict[argument_position][mark_latitude] = attribute_value

    gcp_args_arr = []

    for gcp_args in gcp_args_dict.values():
        args_array = [gcp_args[mark_u], gcp_args[mark_v], gcp_args[mark_longitude], gcp_args[mark_latitude]]
        gcp_args_arr.append(" ".join(args_array))

    return gcp_args_arr


def extract_original_srs(srs_xml_location):
    tree = ETree.parse(srs_xml_location)
    root = tree.getroot()
    wkt = ""
    for element in root.iter("WKT"):
        wkt = element.text

    return wkt.rstrip("\r\n\t")


def geo_reference_raster_map(map_location, gcp_xml_location):
    map_boundaries = "../USGS_24k_Topo_Map_Boundaries_NAD83inNAD27/USGS_24k_Topo_Map_Boundaries_NAD83inNAD27.shp"

    gcp_args_arr = extract_gcp_coordinates(gcp_xml_location)

    gcp_tif = map_location.replace("_orig.tif", "_gcp.tif")
    call_params = ["-of GTiff"] + ["-gcp " + gcp_coords for gcp_coords in gcp_args_arr] + \
                  ["-a_srs NAD27"] + [map_location] + [gcp_tif]

    gdal_translate(*call_params)

    # Get geo-referenced map's SRS
    call_params = ["-o wkt", gcp_tif]
    transform_from = gdalsrsinfo(*call_params).rstrip("\r\n\t")

    # Get SRS into which we need to transform original
    # srs_xml = map_location + ".aux.xml"
    # transform_to = extract_original_srs(srs_xml)

    call_params = ["-o wkt", map_boundaries]
    transform_to = gdalsrsinfo(*call_params).rstrip("\r\n\t")

    # Re-project into quadrant's projection
    outfile = gcp_tif.replace("_gcp.tif", "_geo.tif")
    call_params = ["-wm 600", "-multi", "-order 2", "-r near", "-srcnodata 0", "-dstnodata 255",
                   "-t_srs '" + transform_to + "'", "-s_srs '" + transform_from + "'",
                   gcp_tif, outfile]

    gdalwarp(*call_params)

    # Cleanup after ourselves
    try:
        print "Cleaning up..."
        os.remove(gcp_tif)
    except OSError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geo-reference raster map')
    parser.add_argument("rasterized_map", help='location of rasterized map')
    parser.add_argument("gcp_xml", help='xml file containing gcp coordinates for the rasterized map')

    args = parser.parse_args()

    rasterized_map = args.rasterized_map
    gcp_xml = args.gcp_xml

    geo_reference_raster_map(rasterized_map, gcp_xml)
