import argparse
import os
import subprocess

dirname = "raw_data"
raster_name = "CA_Bray_300000_1988_24000_bag"

working_dir = os.path.join(dirname, raster_name)

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

print working_dir
