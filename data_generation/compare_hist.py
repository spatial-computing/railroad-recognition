import cv2
import numpy as np


# to compare and
# list = {"100.txt","200.txt","199.txt"}
# data_neg = np.loadtxt("../output/test.txt", dtype=int, delimiter=',')
# for item in list:
#     temp = np.loadtxt("../output/" + item, dtype=int, delimiter=',')
#     print item, temp.shape
#     print data_neg.shape
#     data_neg = np.concatenate((temp, data_neg))
path="C:\Users\Vinil\PycharmProjects\Final_data_generation\output"
data_neg = np.loadtxt(path+"\\negative_coordinates1.txt", dtype=int, delimiter=',')

data_pos = np.loadtxt(path+"\\positive_coordinates1.txt", dtype=int, delimiter=',')

data = np.concatenate((data_neg, data_pos))
print data.shape

map_geo = cv2.imread("D:\\data\\CA_Bray_100414_2001_24000_geo.tif")

no_of_bins = 64

a = data[0]
y = int(a[0])
x = int(a[1])
img = map_geo[y - 24:y + 24, x - 24:x + 24]
final_hist = cv2.calcHist([img], [0, 1, 2], None, [no_of_bins, no_of_bins, no_of_bins],
                          [0, 256, 0, 256, 0, 256])
print final_hist.shape

for a in data:
    y = int(a[0])
    x = int(a[1])
    img = map_geo[y - 24:y + 24, x - 24:x + 24]
    hist = cv2.calcHist([img], [0, 1, 2], None, [no_of_bins, no_of_bins, no_of_bins],
                        [0, 256, 0, 256, 0, 256])
    final_hist = np.add(final_hist, hist)

map_hist = cv2.calcHist([map_geo], [0, 1, 2], None, [no_of_bins, no_of_bins, no_of_bins],
                        [0, 256, 0, 256, 0, 256])

map_hist = cv2.normalize(map_hist).flatten()
final_hist = cv2.normalize(final_hist).flatten()

OPENCV_METHODS = (
    ("Correlation", cv2.cv.CV_COMP_CORREL),
    ("Chi-Squared", cv2.cv.CV_COMP_CHISQR),)

for (methodName, method) in OPENCV_METHODS:
    d = cv2.compareHist(map_hist, final_hist, method)
    print methodName, d
