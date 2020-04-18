import subprocess
#full path to gdal executables>
gdalsrsinfo = r'C:\OSGeo4W64\bin\gdalsrsinfo.exe'
ogr2ogr = r'C:\OSGeo4W64\bin\ogr2ogr.exe'

def reproject(source_file, target_file, target_proj):
    call = gdalsrsinfo+' -o proj4 "'+source_file+'"'
    crs_source = subprocess.check_output(call, shell=True).strip().replace("'","")
    print (crs_source)
    tmpt = ' -t_srs {p:s} -s_srs "'+crs_source+'" "'+target_file+'" "'+source_file+'"'
    call = ogr2ogr+tmpt.format(p=target_proj)
    print (call)
    response=subprocess.check_output(call, shell=True)
    print response

source_path = 'C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_bag\data\Perfect_shp\CA_Bray_railroads_2001_perfect_backup.shp'
target_path = source_path.replace('.shp', '_4269.shp')
reproject(source_path, target_path, 'EPSG:4269')
