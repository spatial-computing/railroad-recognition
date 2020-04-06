# evaluation metrics

import numpy as np
import os
    
os.environ["CUDA_VISIBLE_DEVICES"]="0"
TARGET_SAMPLE_DIR = "../filtered"
GT_FILE = './data/bray_2001/bray_gt_v2.txt'
buff = 10

x, target_name = load_data.load_wetland_samples(TARGET_SAMPLE_DIR)
gt = np.loadtxt(GT_FILE, dtype='int32', delimiter=',')
tp, fp, fn = 0.0,0.0,0.0
# img = cv2.imread(MAP_PATH)
p = []
for i in target_name:
    if 'orig' in i or '_' not in i:
        continue
    tmpt = i.split('_')
    r, c = int(tmpt[0]), int(tmpt[1][:-4])
    p.append([r, c])
#     print(r,c)
    flag = 0
    for j in gt:
        if abs(r-j[0]) <= buff and abs(c-j[1]) <= buff:
            tp += 1.0
            flag = 1
            break
    if flag == 0:
        fp += 1.0
for j in gt:
    for i in p:
        if abs(i[0]-j[0]) <= buff and abs(i[1]-j[1]) <= buff:
            fn += 1
            break
print('tp, fp, fn, gt: ', tp, fp,fn, len(gt))
precision = tp / (tp+fp)
recall = fn / len(gt)
f1 = 2*(precision*recall) / (precision+recall)
print('precision, recall, f1 =  ', precision, recall, f1)
