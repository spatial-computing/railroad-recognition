import argparse
import os
from helper_methods import ogr2ogr

# How to run:
# python prepare_vector_data.py quadrangle_state quadrangle_area


def prepare_vector_data(working_dir, quadrangle_state, quadrangle_area):
    map_boundaries = "../USGS_24k_Topo_Map_Boundaries_NAD83inNAD27/USGS_24k_Topo_Map_Boundaries_NAD83inNAD27.shp"
    vectorized_railroads = r'../Trans_RailFeature/Trans_RailFeature.shp'

    # use USGS quadrangle geometry to clip vector exactly to map area
    # first select quadrangle
    quad_select = working_dir + os.sep + 'quadr_' + quadrangle_area + '_' + quadrangle_state + '.shp'
    ogr2ogr_args = ["""-where "QUAD_NAME='%s' AND ST_NAME1='%s'" """ % (quadrangle_area, quadrangle_state),
                    quad_select,
                    map_boundaries]

    response = ogr2ogr(*ogr2ogr_args)
    print response

    # Clip
    output = os.path.join(working_dir, "Trans_RailFeature_clip.shp")
    ogr2ogr_args = ['-dim 2', '-clipsrc ' + quad_select, output, vectorized_railroads]

    return ogr2ogr(*ogr2ogr_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clip vectorized railroads to quadrangle and adjust the path')
    parser.add_argument("working_dir", help='Working Directory')
    parser.add_argument("quadrangle_state", help='Quadrangle State')
    parser.add_argument("quadrangle_area", help='Quadrangle Area')

    args = parser.parse_args()

    prepare_vector_data(args.working_dir, args.quadrangle_state, args.quadrangle_area)
