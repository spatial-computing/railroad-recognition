import cv2
import random
import sys
def is_safe(x,y):
    if y>start_y+24 and  y <end_y-24 and x>start_x+24 and x<end_x-24 and img_perfect[y,x]<235:
        return True
    return False

map_geo=cv2.imread(r"D:\Raster Maps\2001\maps\CA_Bray_100414_2001_24000_bag\data\CA_Bray_100414_2001_24000_geo.tif")

m=r'C:\Users\Vinil\Desktop\vinil\temp\quality_cal\matched_ref'
map=m+'.tif'
map_compressed=m+'.png'

coordinates_file=open("bounding_coordinates.txt","r")
img_perfect=cv2.imread(map_compressed,0)
img_orig=cv2.imread("original0.png",0)
width=img_perfect.shape[1]
height=img_perfect.shape[0]
counts=dict()
count2=dict()
step=1
total=0
outfile_pos=open("positive_coordinates_mix.txt","w")
outfile_neg=open("negative_coordinates_mix_sampled.txt","w")


start_x=int(coordinates_file.readline())
start_y=int(coordinates_file.readline())

end_x=int(coordinates_file.readline())
end_y=int(coordinates_file.readline())

print start_x,start_y,end_x,end_y
coordinates_file.close()


for y in range(start_y+24,end_y-24,step):
    for x in range(start_x,end_x-24,step):
        if img_perfect[y,x]>0:
            total+=1
            i=img_perfect[y,x]
            count2[i]=count2.get(i,0)+1
            if(counts.get(i)==None):
                counts[i]=[[y,x]]
            else:
                counts[i].append([y,x])#=counts.get(i,0)+1

# for key in counts:
#     print len(counts[key])
#     print counts[key]
# outfile=open("positive_coordinates.txt")
# for y in  range(24,height-24,step):
#     for x in range(24,width-24,step):
#         if img_orig[y,x]==255:
#             total+=1
#             i=img_perfect[y,x]
#             count2[i]=count2.get(i,0)+1
            # if(counts.get(i)==None):
            #     counts[i]=[[y,x]]
            # else:
            #     counts[i].append([y,x])#=counts.get(i,0)+1
pos=0
# print counts[100]
val=0
#sys.exit(0)
# print counts[200]
for key in counts:
    if key==100:
        for entry in counts[100]:
             for x in range(entry[0]-250,entry[0]+250,24):
                for y in range(entry[1]-250,entry[1]+250,24):
                    if is_safe(x,y):
                        outfile_neg.writelines(str(y)+","+str(x)+"\n")
    if key==199:
        # counts[key]=random.sample(counts[key],2500)
        print key,len(counts[key])
        pos=0
        step_water=30
        print "start"
        for entry in counts[key]:
            outfile_neg.writelines(str(entry[0])+","+str(entry[1])+"\n")
            # x=entry[1]
            # y=entry[0]
            # new_img=map_geo[y-24:y+24,x-24:x+24]
            # pos+=1
            # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)
            continue
            x=entry[1]+step_water
            y=entry[0]
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]-step_water
            y=entry[0]
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]
            y=entry[0]-step_water
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]
            y=entry[0]+step_water
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)
    elif key==200:
        counts[key]=random.sample(counts[key],2500)
        print key,len(counts[key])
        pos=0
        step_road=15
        for entry in counts[key]:
            outfile_neg.writelines(str(entry[0])+","+str(entry[1])+"\n")
            x=entry[1]
            y=entry[0]
            new_img=map_geo[y-24:y+24,x-24:x+24]
            pos+=1
            # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]+step_road
            y=entry[0]
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]-step_road
            y=entry[0]
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]
            y=entry[0]-step_road
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)

            x=entry[1]
            y=entry[0]+step_road
            if is_safe(x,y):
                pos+=1
                outfile_neg.writelines(str(y)+","+str(x)+"\n")
                # new_img=map_geo[y-24:y+24,x-24:x+24]
                # cv2.imwrite("temp_data/neg_road_img"+str(pos)+".tif",new_img)
    elif(key==255):
        val=0.1
    elif key>=248:
        val=0.1
    elif key>=235:
        val=20/1300.0
    no=len(counts[255])*val
    counts[key]=random.sample(counts[key],int(no))
    print key,len(counts[key])
    for entry in counts[key]:
        outfile_pos.writelines(str(entry[0])+","+str(entry[1])+"\n")
        #print entry[0],entry[1]

outfile_pos.close()
outfile_neg.close()

print total

