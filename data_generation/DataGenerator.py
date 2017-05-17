"""
Copyright 2017 USC SPatial sciences institue

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import cv2
import subprocess
import sys
input_data={}
import gdal
import json
from gdalconst import *
from xml.etree import ElementTree as ET
import glob
from osgeo import ogr
from osgeo import osr
import random
import numpy as np

class Data_generator():
    input_data={}
    vector_info={}

    def __init__(self, config_path):
        self.setup_directories(config_path)

    def getkey_value(self,key):
        return input_data[key]


    def setup_directories(self,config_file_path=""):
        """
        This function will create the output directory(the final output files)
        and a temp directory
        """
        config_data=open(config_file_path)
        json_data=json.load(config_data)
        for key,val in json_data.items():
            input_data[key]=val

        print "Setting up output directories"
        temp="temp"
        if not os.path.exists(temp):
            os.makedirs(temp)
        temp=input_data["output_path"]
        if not os.path.exists(temp):
            os.makedirs(temp)


    def generate_bounding_shp(self,westbc,eastbc,northbc,southbc):

        source = osr.SpatialReference()
        source.ImportFromEPSG(4267)

        target = osr.SpatialReference()
        target.ImportFromEPSG(4269)

        transform = osr.CoordinateTransformation(source, target)
        point = ogr.CreateGeometryFromWkt("POINT ("+westbc +" "+ northbc+" )")
        point.Transform(transform)

        westbc_nad83=str(point.GetX())
        northbc_nad83=str(point.GetY())
        point = ogr.CreateGeometryFromWkt("POINT ("+str(eastbc) +" "+ southbc+" )")
        point.Transform(transform)

        eastbc_nad83=str(point.GetX())
        southbc_nad83=str(point.GetY())

        geo_json='{"type": "FeatureCollection", "features": [{"geometry": {"type": "Polygon", "coordinates": [[['+\
                westbc_nad83+","+northbc_nad83+"],["+eastbc_nad83+","+northbc_nad83+"],["+\
                eastbc_nad83+","+southbc_nad83+"],["+westbc_nad83+","+southbc_nad83+"],["+\
                westbc_nad83+","+northbc_nad83+']]]}, "type": "Feature"}]}'

        out=open("temp/bouding_coordinatesnad83.json","w")
        input_data["bouding_bouding_coordinatesnad83.json"]="temp/bouding_coordinatesnad83.json"
        out.write(geo_json)
        out.close()



    def generate_bounding_coordinates(self):
        """
         this function will output the extent of the map in the bounding_coordinates.txt
        """
        check_keys=["map_metadata","gdal_path","map_path"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists1s "+str(check_keys)
            sys.exit(0)

        bounding_cooridnates_filename=input_data["output_path"]+"bounding_coordinates.txt"
        output_file=open(bounding_cooridnates_filename,"w")
        xml= ET.parse(input_data["map_metadata"])
        westbc=xml.find("./idinfo/spdom/bounding/westbc").text
        eastbc=xml.find("./idinfo/spdom/bounding/eastbc").text
        northbc=xml.find("./idinfo/spdom/bounding/northbc").text
        southbc=xml.find("./idinfo/spdom/bounding/southbc").text


        dataset = gdal.Open( input_data["map_path"], GA_ReadOnly )

#        print westbc,eastbc,northbc,southbc # coordinates in nad23,convert them
        self.generate_bounding_shp(westbc,eastbc,northbc,southbc)

        source = osr.SpatialReference()
        source.ImportFromEPSG(4267)

        target = osr.SpatialReference()
        target.ImportFromWkt(dataset.GetProjection())

        transform = osr.CoordinateTransformation(source, target)
        point = ogr.CreateGeometryFromWkt("POINT ("+eastbc +" "+ southbc+" )")
        point.Transform(transform)
        eastbc_geo=str(point.GetX())
        southbc_geo=str(point.GetY())

        point = ogr.CreateGeometryFromWkt("POINT ("+westbc +" "+ northbc+" )")
        point.Transform(transform)
        westbc_geo=str(point.GetX())
        northbc_geo=str(point.GetY())

        adfGeoTransform = dataset.GetGeoTransform()

        dfGeoX=float(westbc_geo)
        dfGeoY =float(northbc_geo)
        det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4];
        X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
        adfGeoTransform[3]) * adfGeoTransform[2]) / det
        Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
        adfGeoTransform[0]) * adfGeoTransform[4]) / det

        output_file.writelines(str(int(X))+"\n")  #start_x
        output_file.writelines(str(int(Y))+"\n")  #start_y

        dfGeoX=float(eastbc_geo)
        dfGeoY =float(southbc_geo)
        det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4];
        X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
        adfGeoTransform[3]) * adfGeoTransform[2]) / det
        Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
        adfGeoTransform[0]) * adfGeoTransform[4]) / det


        output_file.writelines(str(int(X))+"\n")  #end_x
        output_file.writelines(str(int(Y))+"\n")  #end_y

        output_file.close()
        input_data["bounding_coordinates_file"]=bounding_cooridnates_filename
        print "Bounding Coordinates Generated , location= "+bounding_cooridnates_filename

    def create_blank_raster(self):
        """
        this function will create an empty raster which will be used to rasterise vector data by other functions
        """
        check_keys=["map_path"]

        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)

        map=gdal.Open(input_data["map_path"])
        x_min, pixelSizeX, xskew, y_min, yskew, pixelSizeY  = map.GetGeoTransform()
        NoData_value = 0
        x_max = x_min + (map.RasterXSize *pixelSizeX)
        y_max = y_min + (map.RasterYSize * pixelSizeY)

        # # # Create the destination data source
        y_res = int((y_max - y_min) / pixelSizeY)
        x_res = int((x_max - x_min) / pixelSizeX)

        output_raster=input_data["output_path"]+"raster_data.tif"

        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
        target_ds.SetProjection(map.GetProjectionRef())
        a,b,c,d,e,f=map.GetGeoTransform()
        target_ds.SetGeoTransform((x_min, pixelSizeX, xskew, y_min, yskew, pixelSizeY))#x_min, pixelSizeX, 0, y_max, 0, -pixelSizeY))
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(NoData_value)

        input_data["rasterised_vdata"]=output_raster

        print "created blank raster =  "+output_raster

    def rasterize_single_vector(self,vector_path,color,buffer=0,output_png_path=""):
        """
        this function  will rasterise a single vector on the raster_data.tif with color and buffer as specified
        #if output_png_path is empty, the function wont create the final image, so if you want to rasterize multilpe vector,
        # pass the path in the last call
        """
        check_keys=["gdal_path"]

        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)

        if "rasterised_vdata" not in input_data.keys():
            self.create_blank_raster()

        vector=self.clip_proj(vector_path)
        feature_name=self.extract_feature_name(vector)
        if buffer!=0:
            new_vector="temp/"+feature_name+".shp"
            ogr2ogr = input_data["gdal_path"]+'ogr2ogr'
            call = """%s -f "ESRI Shapefile" %s %s -dialect sqlite -sql "SELECT ST_Union(ST_buffer(Geometry,%f)) FROM '%s'" """ % (ogr2ogr, new_vector, vector,buffer+0.5, feature_name)
            response=subprocess.check_output(call, shell=True)
            print response
            vector=new_vector


        gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize"
        call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(color),feature_name,vector,input_data["rasterised_vdata"])
        response=subprocess.call(call, shell=True)
        print "rasterized "+feature_name+" with value= "+str(color)+ "buffer = "+ str(buffer)

        if output_png_path != "":
            call=input_data["gdal_path"]+"gdal_translate"+" -of PNG %s %s"%(input_data["rasterised_vdata"],output_png_path)
            subprocess.check_output(call, shell=True)
            print "created rasterised vector data = "+output_png_path


    def clip_proj(self,vector_orig):
        """
        This function will clip a vector to the extent of the map.
        If a vector was generated manually, there is no need to clip it. SUch files should be have manual in their name
        """
        # if "manual_" in vector_orig or "clip_proj" in vector_orig:
        if "clip_" in vector_orig or "clip_proj" in vector_orig:
            return vector_orig
        check_keys=["gdal_path","map_path"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        print "start Clipping vector "+vector_orig
        map_tiff_geo = input_data["map_path"]
        gdal=input_data["gdal_path"]
        #full path to gdal executables>
        gdalsrsinfo = gdal+'gdalsrsinfo'
        ogr2ogr = gdal+'ogr2ogr'

        call = gdalsrsinfo+' -o proj4 "'+vector_orig+'"'
        crs_vector=subprocess.check_output(call, shell=True).strip().replace("'","")

        if "crs_raster" in input_data.keys():
            crs_raster=input_data["crs_raster"]
        else :
            call = gdalsrsinfo+' -o proj4 "'+map_tiff_geo+'"'
            crs_raster=subprocess.check_output(call, shell=True).strip().replace("'","")
            input_data["crs_raster"]=crs_raster
        if crs_vector==crs_raster:
            print "not clipping the vector"
            return vector_orig
        #clip
        vector_clip=vector_orig.replace('.shp','_clip.shp')
        bouding_polygon=input_data["bouding_bouding_coordinatesnad83.json"]
        call = '%s -dim 2 -clipsrc %s %s %s ' % (ogr2ogr, bouding_polygon, vector_clip, vector_orig)
        response=subprocess.check_output(call, shell=True)
        print call
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

    def extract_feature_name(self,vector_fullpath):
        feature_name= os.path.basename(vector_fullpath)
        return feature_name.replace(".shp","")

    def rasterize_data(self):
        """
        this function will rasterize all the vector data from roads, railroads, water and mountain peaks folder using
        the value provided in the index
        """
        check_keys=["gdal_path","map_path"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        if "rasterised_vdata" not in input_data.keys():
            print "Missing blank raster,make sure you have called create_blank_raster_api"
            sys.exit(0)


        # vectors_negative=input_data["training_vector_negative"]
        # for item in vectors_negative:
        #     path=item["path"]
        #     all_vectors=self.list_all_vectors_helper(path)
        #     color=item["rasterise_value"]
        #     offset=item["offset_in_pixels"]
        #     a=[offset,item["type"],"negative"]
        #     self.vector_info[int(color)]=[offset,item["type"],"negative"]
        #     for vector_item in all_vectors:
        #         try:
        #             #vector=self.clip_proj(vector_item)
        #             if offset>0 and type!="point":
        #                 self.rasterize_single_vector(vector_item,1,24)
        #         except:
        #             raise
        #
        #             print "!!!!failed "+ vector_item
        #
        # vector_positive=input_data["training_vector_positive"]
        # for item in vector_positive:
        #     path=item["path"]
        #     all_vectors=self.list_all_vectors_helper(path)
        #     color=item["rasterise_value"]
        #     offset=item["offset_in_pixels"]
        #     self.vector_info[int(color)]=[int(offset),item["type"],"positive"]
        #     for vector_item in all_vectors:
        #         try:
        #             vector=self.clip_proj(vector_item)
        #             if offset>0:
        #                 self.rasterize_single_vector(vector,1,24)
        #         except:
        #             print "!!!!failed "+ vector_item

        vectors_negative=input_data["training_vector_negative"]
        for item in vectors_negative:
            path=item["path"]
            all_vectors=self.list_all_vectors_helper(path)
            color=item["rasterise_value"]
            offset=item["offset_in_pixels"]
            limit_samples=item["max_number_of_samples"]
            self.vector_info[int(color)]=[offset,item["type"],"negative",limit_samples]
            for vector_item in all_vectors:
                try:
                    vector=self.clip_proj(vector_item)
                    vector_feature_name=self.extract_feature_name(vector)
                    gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize"
                    call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(color),vector_feature_name,vector,input_data["rasterised_vdata"])
                    response=subprocess.call(call, shell=True)
                    print "rasterized "+vector_feature_name+" with value="+color
                except:

                    print "!!!!failed "+ vector_item

        vector_positive=input_data["training_vector_positive"]
        for item in vector_positive:
            path=item["path"]
            all_vectors=self.list_all_vectors_helper(path)
            color=item["rasterise_value"]
            offset=item["offset_in_pixels"]
            limit_samples=item["max_number_of_samples"]
            self.vector_info[int(color)]=[int(offset),item["type"],"positive",limit_samples]
            for vector_item in all_vectors:
                try:
                    vector=self.clip_proj(vector_item)
                    vector_feature_name=self.extract_feature_name(vector)
                    gdal_rasterize=input_data["gdal_path"]+"gdal_rasterize"
                    call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(int(color),vector_feature_name,vector,input_data["rasterised_vdata"])
                    print call
                    response=subprocess.call(call, shell=True)
                    print "rasterized "+vector_feature_name+" with value="+color
                except:
                    print "!!!!failed "+ vector_item

        input_data["map_compressed"]=input_data["output_path"]+"raster_data.png"
        try:
            call=input_data["gdal_path"]+"gdal_translate"+" -of PNG %s %s"%(input_data["rasterised_vdata"],input_data["map_compressed"])
            subprocess.check_output(call, shell=True)
            print "created rasterised vector data"+input_data["map_compressed"]

        except:
            print "failed creating rasterised vector data...exiting"
            sys.exit(0)

    def generate_testing_data(self):
        check_keys=["gdal_path","map_path","no_of_buffers_testing","testing_ground_truth","step_testing","window_size"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        buffer=int(self.getkey_value("no_of_buffers_testing"))

        for i in range(buffer,0,-1):
            self.rasterize_single_vector(self.getkey_value("testing_ground_truth"),255-i,i)

        test_image_path=input_data["output_path"]+"test.png"
        self.rasterize_single_vector(self.getkey_value("testing_ground_truth"),255,0,test_image_path)

        step=int(self.getkey_value("step_testing"))

        coordinates_file=open(input_data["output_path"]+"bounding_coordinates.txt","r")

        start_x=int(coordinates_file.readline())
        start_y=int(coordinates_file.readline())

        end_x=int(coordinates_file.readline())
        end_y=int(coordinates_file.readline())

        outfiles=[]
        for i in range(0,buffer+1):
            path=input_data["output_path"]+"testing_positive_coordinates_buffer"+str(i)+".txt"
            file=open(path,"w")
            outfiles.append(file)

        window_sizeby2=int(self.getkey_value("window_size"))/2
        img_orig=cv2.imread(test_image_path,0)

        print "saving locations in output folder, Dont Quit!!"
        for y in range(start_y+window_sizeby2,end_y-window_sizeby2,step):
                    for x in range(start_x,end_x-window_sizeby2,step):
                        for i in range(0,buffer+1):
                            if img_orig[y,x]>=255-i:
                                outfiles[i].writelines(str(y)+","+str(x)+"\n");

        for f in outfiles:
            f.close()

    def generate_positive_negative_coordinates(self):
        print "generate_positive_negative_coordinates2"

        if not "map_compressed" in input_data.keys():
            print "Rasterize data first"
            sys.exit(0)
        check_keys=["step_training","window_size"]
        if not set(check_keys).issubset(set(input_data.keys())):
            print "missing entries in config file, Please if these exists "+str(check_keys)
            sys.exit(0)
        coordinates_file=open(input_data["output_path"]+"bounding_coordinates.txt","r")

        start_x=int(coordinates_file.readline())
        start_y=int(coordinates_file.readline())

        end_x=int(coordinates_file.readline())
        end_y=int(coordinates_file.readline())

        coordinates_file.close()
        if "step_training" not in input_data.keys():
            print "step training not found in config...Using default=1"
            step=1
        else:
            step=int(input_data["step_training"])

        print "generating positive and negative coordinates,Dont Quit"
        counts=dict()
        count2=dict()
        img_perfect=cv2.imread(input_data["map_compressed"],0)

        if "window_size" not in input_data.keys():
            print "window_size not found...exiting"
            sys.exit(0)

        window_sizeby2=int(input_data["window_size"])/2

        for y in range(start_y+window_sizeby2,end_y-window_sizeby2,step):
            for x in range(start_x,end_x-window_sizeby2,step):
               if img_perfect[y,x]>0:
                    i=img_perfect[y,x]
                    count2[i]=count2.get(i,0)+1
                    if(counts.get(i)==None):
                        counts[i]=[[y,x]]
                    else:
                        counts[i].append([y,x])
        outfile_pos=open(input_data["output_path"]+"positive_coordinates.txt","w")
        outfile_neg=open(input_data["output_path"]+"negative_coordinates.txt","w")

        no_neg_samples=0
        for key in counts:
            if key==0:
                continue
            samples=[]

            vector_info_item=self.vector_info[key]
            offset=int(vector_info_item[0])
            max_number_of_samples=int(vector_info_item[3])
            is_neg=1
            if vector_info_item[2]=="positive":
                out=outfile_pos
                is_neg=0
            else: out=outfile_neg
            type=vector_info_item[1]
            if type=="point":
                step=4
                for entry in counts[key]:
                     for y in range(entry[0]-offset/2,entry[0]+offset/2,step):
                        for x in range(entry[1]-offset/2,entry[1]+offset/2,step):
                            if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=255:
                                samples.append([y,x])
                                #out.writelines(str(y)+","+str(x)+"\n")
                                if is_neg:
                                    no_neg_samples+=1

            elif vector_info_item[1]=="line":

                pos=0
                for entry in counts[key]:
                    #out.writelines(str(entry[0])+","+str(entry[1])+"\n")
                    samples.append([str(entry[0]),str(entry[1])])
                    if is_neg:
                        no_neg_samples+=1
                    if offset==0:
                        continue
                    x=entry[1]+offset
                    y=entry[0]
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=255:
                        pos+=1
                        samples.append([y,x])
                        #out.writelines(str(y)+","+str(x)+"\n")
                        if is_neg:
                            no_neg_samples+=1

                    x=entry[1]-offset
                    y=entry[0]
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=255:
                        pos+=1
                        #out.writelines(str(y)+","+str(x)+"\n")
                        samples.append([y,x])
                        if is_neg:
                            no_neg_samples+=1

                    x=entry[1]
                    y=entry[0]-offset
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=255:
                        pos+=1
                        # out.writelines(str(y)+","+str(x)+"\n")
                        samples.append([y,x])
                        if is_neg:
                            no_neg_samples+=1

                    x=entry[1]
                    y=entry[0]+offset
                    if y>start_y+window_sizeby2  and  y <end_y-window_sizeby2 and x>start_x+window_sizeby2 \
                                    and x<end_x-window_sizeby2 and img_perfect[y,x]!=255:
                        pos+=1
                        # out.writelines(str(y)+","+str(x)+"\n")
                        samples.append([y,x])
                        if is_neg:
                            no_neg_samples+=1
            else:
                for entry in counts[key]:
                    # out.writelines(str(entry[0])+","+str(entry[1])+"\n")
                    samples.append([str(entry[0]),str(entry[1])])
                    if is_neg:
                        no_neg_samples+=1

            if max_number_of_samples>0 and max_number_of_samples< len(samples):
                samples=random.sample(samples,max_number_of_samples)
            individual_feature_data=open(input_data["output_path"]+str(key)+".txt","w")
            for item in samples:
                out.writelines(str(item[0])+","+str(item[1])+"\n")
                individual_feature_data.writelines(str(item[0])+","+str(item[1])+"\n")
            individual_feature_data.close()
            max_number_of_samples=-1
            samples=[]

        #the following code can be used to generate random samples

        target=int(input_data["number_of_random_samples"])
        individual_feature_data=open(input_data["output_path"]+str("random")+".txt","w")
        i=0
        while i<target:
            x=random.randint(start_x,end_x)
            y=random.randint(start_y,end_y)
            if img_perfect[y,x]==0:
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                individual_feature_data.writelines(str(item[0])+","+str(item[1])+"\n")
                i+=1

        outfile_pos.close()
        outfile_neg.close()

        print "Positive Coordinates = " + input_data["output_path"]+"positive_coordinates.txt"
        print "negative Coordinates = " + input_data["output_path"]+"negative_coordinates.txt"
        self.plot_samples()
        return

    def plot_samples(self):

        data_pos=np.loadtxt(input_data["output_path"]+"positive_coordinates1.txt",dtype=int,delimiter=',')
        data_neg=np.loadtxt(input_data["output_path"]+"negative_coordinates1.txt",dtype=int,delimiter=',')

        map_geo=cv2.imread(input_data["map_path"])
        output=map_geo.copy()
        for item in data_pos:
            cv2.circle(map_geo,(item[1],item[0]),11, (0,0,255), -1)
        for item in data_neg:
            cv2.circle(map_geo,(item[1],item[0]),11, (255,0,0), -1)


        alpha=0.5
        cv2.addWeighted(map_geo, alpha, output, 1 - alpha,
                0, output)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, encimg = cv2.imencode('.jpg', output, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        cv2.imwrite(input_data["output_path"]+"plot_samples.jpg",decimg)

        print "samples plotted in "+input_data["output_path"]+"plot_samples.jpg"