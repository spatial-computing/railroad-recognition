import os
import cv2
import subprocess
import sys
input_data={}
import gdal

from gdalconst import *
from xml.etree import ElementTree as ET
import glob

class Training_Data_generator():
    input_data={}
    def __init__(self, config_path):
        self.setup_directories(config_path)

    def getkey_value(self,key):
        return input_data[key]


    def setup_directories(self,config_file_path):
        print "Setting up output directories"
        output="output"
        temp="temp"
        if not os.path.exists(output):
            os.makedirs(output)
        if not os.path.exists(temp):
            os.makedirs(temp)
        with open(config_file_path) as f:
            for line in f:
               line=line.strip()
               try:
                (key, val) = line.split('=')
               except:
                   print line
               input_data[key] = val

    def generate_bounding_coordinates(self):
        check_keys=["map_metadata","gdal_path","map_path"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists1s "+str(check_keys)
            sys.exit(0)

        bounding_cooridnates_filename="output\\bounding_coordinates.txt"
        output_file=open(bounding_cooridnates_filename,"w")
        xml= ET.parse(input_data["map_metadata"])
        westbc=xml.find("./idinfo/spdom/bounding/westbc").text
        eastbc=xml.find("./idinfo/spdom/bounding/eastbc").text
        northbc=xml.find("./idinfo/spdom/bounding/northbc").text
        southbc=xml.find("./idinfo/spdom/bounding/southbc").text

        #print westbc,eastbc,northbc,southbc # coordinates in nad23,convert them

        if "crs_raster" in input_data.keys():
            crs_raster=input_data["crs_raster"]
        else:
            call = input_data["gdal_path"]+"gdalsrsinfo.exe" +' -o proj4 "'+input_data["map_path"]+'"'
            crs_raster=subprocess.check_output(call, shell=True).strip().replace("'","")
            input_data["crs_raster"]=crs_raster

        call2=input_data["gdal_path"]+'gdaltransform.exe  -t_srs "'+crs_raster+ '" -s_srs EPSG:4267'
        #print call2

        p = subprocess.Popen(call2,stdout=subprocess.PIPE,stdin=subprocess.PIPE)

        c=eastbc+" "+southbc

        p.stdin.write(c)
        temp=p.communicate()[0].split(" ")
        eastbc_geo=temp[0]
        southbc_geo=temp[1]
        p.stdin.close()

        p = subprocess.Popen(call2,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        c=westbc+" "+northbc
        p.stdin.write(c)
        temp=p.communicate()[0].split(" ")
        westbc_geo=temp[0]
        northbc_geo=temp[1]
        p.stdin.close()


        dataset = gdal.Open( input_data["map_path"], GA_ReadOnly )

        adfGeoTransform = dataset.GetGeoTransform()


        dfGeoX=float(westbc_geo)
        dfGeoY =float(northbc_geo)
        det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4];
        X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
        adfGeoTransform[3]) * adfGeoTransform[2]) / det;
        Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
        adfGeoTransform[0]) * adfGeoTransform[4]) / det;

        output_file.writelines(str(int(X))+"\n")  #start_x
        output_file.writelines(str(int(Y))+"\n")  #start_y

        dfGeoX=float(eastbc_geo)
        dfGeoY =float(southbc_geo)
        det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4];
        X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
        adfGeoTransform[3]) * adfGeoTransform[2]) / det;
        Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
        adfGeoTransform[0]) * adfGeoTransform[4]) / det;


        output_file.writelines(str(int(X))+"\n")  #end_x
        output_file.writelines(str(int(Y))+"\n")  #end_y

        output_file.close()
        input_data["bounding_coordinates_file"]=bounding_cooridnates_filename
        print "Bounding Coordinates Generated , location= "+bounding_cooridnates_filename

    def create_blank_raster(self):

        check_keys=["map_path"]

        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)

        map=gdal.Open(input_data["map_path"])
        x_min, pixelSizeX, xskew, y_min, yskew, pixelSizeY  = map.GetGeoTransform()
        NoData_value = 0

        # Open the data source and read in the extent

        x_max = x_min + (map.RasterXSize *pixelSizeX)
        y_max = y_min + (map.RasterYSize * pixelSizeY)

        # Create the destination data source
        x_res = int((x_max - x_min) / pixelSizeX)
        y_res = int((y_max - y_min) / pixelSizeY)
        output_raster="output/raster_data.tif"
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
        target_ds.SetProjection(map.GetProjectionRef())
        target_ds.SetGeoTransform((x_min, pixelSizeX, 0, y_max, 0, -pixelSizeY))
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(NoData_value)

        input_data["rasterised_vdata"]=output_raster

        print "created blank raster =  "+output_raster

    def rasterize_single_vector(self,vector_path,color,buffer=0,output_png_path=""):

        #if output_png_path is empty, the function wont create the final image, so if you want to rasterize multilpe vector,
        # pass the path in the last call

        if "rasterised_vdata" not in input_data.keys():
            self.create_blank_raster()

        vector=self.clip_proj(vector_path)
        feature_name=self.extract_feature_name(vector)
        if buffer!=0:
            new_vector="temp/"+feature_name+".shp"
            ogr2ogr = input_data["gdal_path"]+'ogr2ogr.exe'
            call = """%s -f "ESRI Shapefile" %s %s -dialect sqlite -sql "SELECT ST_Union(ST_buffer(Geometry,%f)) FROM '%s'" """ % (ogr2ogr, new_vector, vector,buffer+0.5, feature_name)
            print call
            response=subprocess.check_output(call, shell=True)
            print response
            vector=new_vector


        gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize.exe"
        call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(color),feature_name,vector,input_data["rasterised_vdata"])
        response=subprocess.call(call, shell=True)
        print "rasterized "+feature_name+" with value="+str(color)

        if output_png_path != "":
            call=input_data["gdal_path"]+"gdal_translate.exe"+" -of PNG %s %s"%(input_data["rasterised_vdata"],output_png_path)
            subprocess.check_output(call, shell=True)
            print "created rasterised vector data"+output_png_path


    def clip_proj(self,vector_orig):
        if "manual_" in vector_orig or "clip_proj" in vector_orig:
            return vector_orig
        print "start Clipping vector "+vector_orig
        map_tiff_geo = input_data["map_path"]
        gdal=input_data["gdal_path"]
        #full path to gdal executables>
        gdalsrsinfo = gdal+'gdalsrsinfo.exe'
        ogr2ogr = gdal+'ogr2ogr.exe'

        #the USGS quadrangle shapefile:
        quadrangles = input_data["quadrangles"]


        #the name and state of current map quadrangle:
        quadrangle_name = input_data["quadrangle_name"]
        quadrangle_state = input_data["quadrangle_state"]

        #------------------------------------------------------------------------------


        call = gdalsrsinfo+' -o proj4 "'+vector_orig+'"'
        crs_vector=subprocess.check_output(call, shell=True).strip().replace("'","")

        if "crs_raster" in input_data.keys():
            crs_raster=input_data["crs_raster"]
        else :
            call = gdalsrsinfo+' -o proj4 "'+map_tiff_geo+'"'
            crs_raster=subprocess.check_output(call, shell=True).strip().replace("'","")
            input_data["crs_raster"]=crs_raster

        #use USGS quadrangle geometry to clip vector exactly to map area
        #first select quadrangle
        workdir,quads = os.path.split(quadrangles)
        quad_select=workdir+os.sep+'quadr_'+quadrangle_name+'_'+quadrangle_state+'.shp'
        call = """%s -where "QUAD_NAME='%s' AND ST_NAME1='%s'" %s %s""" % (ogr2ogr, quadrangle_name, quadrangle_state, quad_select, quadrangles)
        response=subprocess.check_output(call, shell=True)

        #clip
        vector_clip=vector_orig.replace('.shp','_clip.shp')
        call = '%s -dim 2 -clipsrc %s %s %s ' % (ogr2ogr, quad_select, vector_clip, vector_orig)
        response=subprocess.check_output(call, shell=True)

        # Reproject vector geometry to same projection as raster
        vector_proj = vector_clip.replace('.shp','_proj.shp')
        call = ogr2ogr+' -t_srs "'+crs_raster+'" -s_srs "'+crs_vector+'" "'+vector_proj+'" "'+vector_clip+'"'
        response=subprocess.check_output(call, shell=True)
        print "Clipped vector "+vector_proj
        return vector_proj


    def list_all_vectors_helper(self,directory_path):
        vectors=[]
        list_files=glob.glob(directory_path+"*.shp")
        for item in list_files:
            if not "proj" in item and not "clip" in item:
                vectors.append(item)

        return  vectors


    def extract_feature_name(self,vector):
        k=vector.rfind("\\")
        vector=vector[k+1:]
        vector=vector.replace(".shp","")
        return  vector

    def rasterize_data(self):

        check_keys=["quadrangles","gdal_path","map_path","quadrangle_name","quadrangle_state"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        if "rasterised_vdata" not in input_data.keys():
            print "Missing blank raster,make sure you have called create_blank_raster_api"
            sys.exit(0)

        all_road_vectors=self.list_all_vectors_helper(input_data["road_shapefile"])
        for road_vector_item in all_road_vectors:
            try:
                road_vector=self.clip_proj(road_vector_item)
                road_feature_name=self.extract_feature_name(road_vector)
                gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize.exe"
                call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(input_data["road_color"]),road_feature_name,road_vector,input_data["rasterised_vdata"])
                response=subprocess.call(call, shell=True)
                print "rasterized "+road_feature_name+" with value="+input_data["road_color"]
            except:
                print "!!!!failed "+ road_vector_item


        #sys.exit(0)

        all_water_vectors=self.list_all_vectors_helper(input_data["water_shapefile"])
        for water_vector_item in all_water_vectors:
            try:
                water_vector=self.clip_proj(water_vector_item)
                print water_vector
                water_feature_name=self.extract_feature_name(water_vector)
                print water_feature_name
                gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize.exe"
                call=gdal_rasterize +' -b 1 -burn %d -l %s %s %s'%(int(input_data["water_color"]),water_feature_name,water_vector,input_data["rasterised_vdata"])
                print call
                response=subprocess.check_output(call, shell=True)
                print response
                print "rasterized "+water_feature_name+" with value="+input_data["water_color"]
            except:
                 print "!!!!failed "+ water_vector_item

        all_mountain_vectors=self.list_all_vectors_helper(input_data["mountain_shapefile"])
        for mountain_vector_item in all_mountain_vectors:
            try:
                mountain_vector=self.clip_proj(mountain_vector_item)
                mountain_feature_name=self.extract_feature_name(mountain_vector)
                gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize.exe"
                call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(input_data["mountain_color"]),mountain_feature_name,mountain_vector,input_data["rasterised_vdata"])
                subprocess.check_output(call, shell=True)
                print "rasterized "+mountain_feature_name+" with value="+input_data["mountain_color"]
            except:
                 print "!!!!failed "+ mountain_vector_item

        all_rail_road_vectos=self.list_all_vectors_helper(input_data["railroad_shapefile"])
        for railroad_vector_item in all_rail_road_vectos:
            try:
                railroad_vector=self.clip_proj(railroad_vector_item)
                railroad_feature_name=self.extract_feature_name(railroad_vector)
                gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize.exe"
                call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(input_data["railroad_color"]),railroad_feature_name,railroad_vector,input_data["rasterised_vdata"])
                subprocess.check_output(call, shell=True)
                print "rasterized "+railroad_feature_name+" with value="+input_data["railroad_color"]
            except:
                print "!!!Failed "+ railroad_vector_item


        input_data["map_compressed"]="output\\raster_data.png"
        try:
            call=input_data["gdal_path"]+"gdal_translate.exe"+" -of PNG %s %s"%(input_data["rasterised_vdata"],input_data["map_compressed"])
            subprocess.check_output(call, shell=True)
            print "created rasterised vector data"+input_data["map_compressed"]

        except:
            print "failed creating rasterised vector data...exiting"
            sys.exit(0)

    def generate_testing_data(self):
        check_keys=["gdal_path","map_path","buffer","ground_truth","step_testing","window_size"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        buffer=int(self.getkey_value("buffer"))

        for i in range(buffer,0,-1):
            self.rasterize_single_vector(self.getkey_value("ground_truth"),255-i,i)

        test_image_path="output/test.png"
        self.rasterize_single_vector(self.getkey_value("ground_truth"),255,0,test_image_path)

        step=int(self.getkey_value("step_testing"))

        coordinates_file=open("output/bounding_coordinates.txt","r")

        start_x=int(coordinates_file.readline())
        start_y=int(coordinates_file.readline())

        end_x=int(coordinates_file.readline())
        end_y=int(coordinates_file.readline())

        outfiles=[]
        for i in range(0,buffer+1):
            path="output/testing_positive_coordinates_buffer"+str(i)+".txt"
            file=open(path,"w")
            outfiles.append(file)

        window_sizeby2=int(self.getkey_value("window_size"))/2
        img_orig=cv2.imread(test_image_path,0)

        for y in range(start_y+window_sizeby2,end_y-window_sizeby2,step):
                    for x in range(start_x,end_x-window_sizeby2,step):
                        for i in range(0,buffer+1):
                            if img_orig[y,x]>=255-i:
                                outfiles[i].writelines(str(y)+","+str(x)+"\n");

        for f in outfiles:
            f.close()

    def generate_positive_negative_coordinates(self):

        check_keys=["step_training","map_compressed","window_size"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        coordinates_file=open("output/bounding_coordinates.txt","r")

        start_x=int(coordinates_file.readline())
        start_y=int(coordinates_file.readline())

        end_x=int(coordinates_file.readline())
        end_y=int(coordinates_file.readline())

        coordinates_file.close()
        step=int(input_data["step_training"])
        print "generating positive and negative coordinates,Dont Quit"
        counts=dict()
        count2=dict()
        img_perfect=cv2.imread(input_data["map_compressed"],0)
        window_sizeby2=int(input_data["window_size"])/2
        for y in range(start_y+window_sizeby2,end_y-window_sizeby2,step):
            for x in range(start_x,end_x-window_sizeby2,step):
                if img_perfect[y,x]>0:
                    # total+=1
                    i=img_perfect[y,x]
                    count2[i]=count2.get(i,0)+1
                    if(counts.get(i)==None):
                        counts[i]=[[y,x]]
                    else:
                        counts[i].append([y,x])
        outfile_pos=open("output/positive_coordinates.txt","w")
        outfile_neg=open("output/negative_coordinates.txt","w")

        mcolor=int(input_data["mountain_color"])
        wcolor=int(input_data["water_color"])
        rcolor=int(input_data["road_color"])
        railcolor=int(input_data["railroad_color"])

        for key in counts:
            if key==mcolor:
                mountain_offset=int(input_data["mountain_offset"])
                for entry in counts[mcolor]:
                     for x in range(entry[0]-mountain_offset/2,entry[0]+mountain_offset/2,window_sizeby2):
                        for y in range(entry[1]-mountain_offset/2,entry[1]+mountain_offset/2,window_sizeby2):
                            if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                                outfile_neg.writelines(str(y)+","+str(x)+"\n")
            if key==wcolor:
                # counts[key]=random.sample(counts[key],2500)
                pos=0
                step_water=int(input_data["water_offset"])
                for entry in counts[key]:
                    outfile_neg.writelines(str(entry[0])+","+str(entry[1])+"\n")
                    x=entry[1]+step_water
                    y=entry[0]
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")

                    x=entry[1]-step_water
                    y=entry[0]
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")

                    x=entry[1]
                    y=entry[0]-step_water
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")

                    x=entry[1]
                    y=entry[0]+step_water
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")
            elif key==rcolor:
                #counts[key]=random.sample(counts[key],2500)
                pos=0
                step_road=int(input_data["road_offset"])
                for entry in counts[key]:
                    outfile_neg.writelines(str(entry[0])+","+str(entry[1])+"\n")
                    x=entry[1]
                    y=entry[0]

                    x=entry[1]+step_road
                    y=entry[0]
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")

                    x=entry[1]-step_road
                    y=entry[0]
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")

                    x=entry[1]
                    y=entry[0]-step_road
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")


                    x=entry[1]
                    y=entry[0]+step_road
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=railcolor:
                        pos+=1
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")
            elif(key==railcolor):
                for entry in counts[key]:
                    outfile_pos.writelines(str(-1*entry[0])+","+str(entry[1])+"\n")

        outfile_pos.close()
        outfile_neg.close()

        print "Positive Coordinates = " + "output/positive_coordinates.txt"
        print "negative Coordinates = " + "output/negative_coordinates.txt"







