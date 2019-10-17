import cv2
import numpy as np
import math
import copy
import random
from sklearn.cluster import KMeans
import time
import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor
import os.path
import sys

def color_gradient(map_hsv, p1, p2):
    x_min = min(p1[0], p2[0])
    x_max = max(p1[0], p2[0])
    y_min = min(p1[1], p2[1])
    y_max = max(p1[1], p2[1])
    res = []

    if abs(x_min-x_max) > abs(y_min-y_max):
        k = (p1[1]*1.0-p2[1])/(p1[0]-p2[0]+0.001)
        b = p1[1] - k*p1[0]
        for i in range(x_min, x_max+1):
            res.append(map_hsv[int(i), int(round(k*i+b))])
    else:
        k = (p1[0]*1.0-p2[0])/(p1[1]-p2[1]+0.001)
        b = p1[0] - k*p1[1]
        for i in range(y_min, y_max+1):
            res.append(map_hsv[int(round(k*i+b)),int(i)])

    return res

def reward_kmean(points, feature_color_low, feature_color_high):
    if len(points)<2:
        return 1
    # Number of clusters
    kmeans = KMeans(n_clusters=2)
    # Fitting the input data
    kmeans = kmeans.fit(points)
    # Centroid values
    centroids = kmeans.cluster_centers_
    for c in centroids:
        for d in range(len(c)):
            if c[d] < feature_color_low[d] or c[d] > feature_color_high[d]:
                return -1
    return 1

def interpolation(start, end, inter_dis):
    dis = math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
    segment = []
    if  dis > inter_dis:
        # add_num = round(dis/inter_dis, 0)
        add_num = int(dis/inter_dis)
        x_interval = int(round((end[0]-start[0])/float(add_num)))
        y_interval = int(round((end[1]-start[1])/float(add_num)))
        for i in range(0, int(add_num)):
            segment.append([start[0]+i*x_interval, start[1]+i*y_interval])
    segment.append(end)
    return segment

def grabCut(img, win_size, LB, HB):
    mask = np.zeros(img.shape[:2],np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # if img_gray[i,j] > 150 and img_gray[i,j] < 200:
            if img[i,j,0] > LB[0] and img[i,j,0] < HB[0] and img[i,j,1] > LB[1] and img[i,j,1] < HB[1] and img[i,j,2] > LB[2] and img[i,j,2] < HB[2]:
                mask[i,j] = 1
    mask = mask.astype('uint8')
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,win_size,win_size)
    try:
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        return img
    except cv2.error:
        return np.zeros((win_size,win_size,3)).astype('uint8')

def foreground_indeces(l):
    flag = 0
    foreground_indices = []
    for i in range(len(l)):
        if l[i] == 0 and flag == 1:
            foreground_indices.append(set(tmpt))
            flag = 0
        if l[i] == 1 and flag == 0:
            tmpt = [i]
            flag = 1
        if l[i] == 1 and flag == 1:
            tmpt.append(i)
        if i==len(l)-1 and flag == 1:
            foreground_indices.append(set(tmpt))
    dis = []
    for i in foreground_indices:
        dis.append(list(i)[len(i)/2])
    return dis


def Q_value_initial(foreground, win_size):
    # Q = [[C, N, S, W, E, NW, NE, SW, SE], [], []...], each point is a list
    mask = np.zeros((win_size,win_size))
    mask[win_size/2,:] = 1
    mask[:,win_size/2] = 1
    for i in range(mask.shape[1]):
        mask[i,i] = 1
    for i in range(mask.shape[1]):
        mask[i,win_size-1-i] = 1
    foreground_gray = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
    result = np.logical_and(foreground_gray,mask)
    Q = [[-100],[-100],[-100],[-100],[-100],[-100],[-100],[-100],[-100]]
    distance = [[[win_size,win_size]],[[win_size,win_size]],[[win_size,win_size]],[[win_size,win_size]],[[win_size,win_size]],\
                [[win_size,win_size]],[[win_size,win_size]],[[win_size,win_size]],[[win_size,win_size]]]
    if result[win_size/2,win_size/2] !=0:
        Q[0] = [0]
        distance[0] = [[0,0]]
    # North
    tmpt = result[0:win_size/2-1, win_size/2]
    dis = foreground_indeces(tmpt)
    if len(dis)>0:
        Q[1] = [0]*len(dis)
        distance[1] = [[i-win_size/2,0] for i in dis]

    # South
    tmpt = result[win_size/2+2:win_size,win_size/2]
    dis = foreground_indeces(tmpt)
    if len(dis)>0:
        Q[2] = [0]*len(dis)
        distance[2] = [[i+2,0] for i in dis]

    # West
    tmpt = result[win_size/2,0:win_size/2-1]
    dis = foreground_indeces(tmpt)
    if len(dis)>0:
        Q[3] = [0]*len(dis)
        distance[3] = [[0,i-win_size/2] for i in dis]

    # East
    tmpt = result[win_size/2, win_size/2+2:win_size]
    dis = foreground_indeces(tmpt)
    if len(dis)>0:
        Q[4] = [0]*len(dis)
        distance[4] = [[0,i+2] for i in dis]

    # Northwest
    tmpt = [result[i,i] for i in range(win_size/2-1)]
    dis = foreground_indeces(tmpt)
    if len(dis)>0:
        Q[5] = [0]*len(dis)
        distance[5] = [[i-win_size/2,i-win_size/2] for i in dis]

    # Northeast
    tmpt = [result[i,win_size-i] for i in range(win_size/2+2, win_size)]
    dis = foreground_indeces(tmpt)
    if len(dis)>0:
        Q[6] = [0]*len(dis)
        distance[6] = [[i+2,-(i+2)] for i in dis]

    # Southwest
    tmpt = [result[win_size-1-i,i] for i in range(win_size/2-1)]
    dis = foreground_indeces(tmpt)
    if len(dis) > 0:
        Q[7] = [0]*len(dis)
        distance[7] = [[win_size/2-i,i-win_size/2] for i in dis]

    # Southeast
    tmpt = [result[i,i] for i in range(win_size/2+2, win_size)]
    dis = foreground_indeces(tmpt)
    if len(dis) > 0:
        Q[8] = [0]*len(dis)
        distance[8] = [[i+2,i+2] for i in dis]

    return Q, distance

def choose_action(Q, point):
    # point = random.randint(0, len(Q)-1)
    indices = [i for i, x in enumerate(Q[point]) if x != [-100]]
    a = random.choice(indices)
    sub_indices = [i for i in range(len(Q[point][a]))]
    aa = random.choice(sub_indices)
    return a, aa

def entire_line(map_img, points, feature_color_low, feature_color_high):
    res = []
    for p in range(1, len(points)):
        res.append(reward_kmean(color_gradient(map_img, points[p-1], points[p]), feature_color_low, feature_color_high))
    count_pos = 0.0
    for i in res:
        if i == 1:
            count_pos += 1.0
    return count_pos/len(res)

def greedy(Q_func, state, action, Q, point, record, movement, feature_color_low, feature_color_high):
    prev_p = int(action[:-2])
    if prev_p == 0:
        tmpt_dir1 = int(state[0])
        tmpt_dis1 = int(state[1])
        tmpt_dir2 = int(state[2])
        tmpt_dis2 = int(state[3])
        if tmpt_dir1!= 0:
            tmpt_p1 = [point[0][0]+movement[0][tmpt_dir1][tmpt_dis1][0],point[0][1]+movement[0][tmpt_dir1][tmpt_dis1][1]]
        else:
            tmpt_p1 = [point[0][0], point[0][1]]
        if tmpt_dir2!= 0:
            tmpt_p2 = [point[1][0]+movement[1][tmpt_dir2][tmpt_dis2][0],point[1][1]+movement[1][tmpt_dir2][tmpt_dis2][1]]
        else:
            tmpt_p2 = [point[1][0], point[1][1]]
        if reward_kmean(color_gradient(map_int, tmpt_p1, tmpt_p2), feature_color_low, feature_color_high) == 1:
            return str(1)+state[1*2]+state[1*2+1]
        else:
            for i in range(len(Q[1])):
                if Q[1][i] != [-100]:
                    for j in range(len(Q[1][i])):
                        tmpt_p2 = [point[1][0]+movement[1][i][j][0],point[1][1]+movement[1][i][j][1]]
                        reward = reward_kmean(color_gradient(map_int, tmpt_p1, tmpt_p2), feature_color_low, feature_color_high)
                        if reward == 1:
                            return str(1)+str(i)+str(j)
    elif prev_p == len(Q)-1:
        tmpt_dir1 = int(state[0])
        tmpt_dis1 = int(state[1])
        tmpt_dir2 = int(state[2])
        tmpt_dis2 = int(state[3])
        if tmpt_dir1!= 0:
            tmpt_p1 = [point[0][0]+movement[0][tmpt_dir1][tmpt_dis1][0],point[0][1]+movement[0][tmpt_dir1][tmpt_dis1][1]]
        else:
            tmpt_p1 = [point[0][0], point[0][1]]
        if tmpt_dir2!= 0:
            tmpt_p2 = [point[1][0]+movement[1][tmpt_dir2][tmpt_dis2][0],point[1][1]+movement[1][tmpt_dir2][tmpt_dis2][1]]
        else:
            tmpt_p2 = [point[1][0], point[1][1]]
        if reward_kmean(color_gradient(map_int, tmpt_p1, tmpt_p2), feature_color_low, feature_color_high) != 1:
            a, aa = choose_action(Q, 0)
            return str(0)+str(a)+str(aa)
        else:
            return str(0)+state[:2]
    else:
        tmpt_dir1 = int(state[2*(prev_p)])
        tmpt_dis1 = int(state[2*(prev_p)+1])
        tmpt_dir2 = int(state[2*(prev_p+1)])
        tmpt_dis2 = int(state[2*(prev_p+1)+1])
        if tmpt_dir1!= 0:
            tmpt_p1 = [point[prev_p][0]+movement[prev_p][tmpt_dir1][tmpt_dis1][0],point[prev_p][1]+movement[prev_p][tmpt_dir1][tmpt_dis1][1]]
        else:
            tmpt_p1 = [point[prev_p][0], point[prev_p][1]]
        if tmpt_dir2!= 0:
            tmpt_p2 = [point[prev_p+1][0]+movement[prev_p+1][tmpt_dir2][tmpt_dis2][0],point[prev_p+1][1]+movement[prev_p+1][tmpt_dir2][tmpt_dis2][1]]
        else:
            tmpt_p2 = [point[prev_p+1][0], point[prev_p+1][1]]
        if reward_kmean(color_gradient(map_int, tmpt_p1, tmpt_p2), feature_color_low, feature_color_high) == 1:
            return str(prev_p+1)+state[(prev_p+1)*2]+state[(prev_p+1)*2+1]
        else:
            for i in range(len(Q[prev_p+1])):
                if Q[prev_p+1][i] != [-100]:
                    for j in range(len(Q[prev_p+1][i])):
                        tmpt_p2 = [point[prev_p+1][0]+movement[prev_p+1][i][j][0],point[prev_p+1][1]+movement[prev_p+1][i][j][1]]
                        reward = reward_kmean(color_gradient(map_int, tmpt_p1, tmpt_p2), feature_color_low, feature_color_high)
                        if reward == 1:
                            return str(prev_p+1)+str(i)+str(j)
    p_rand = random.randint(0, len(Q)-1)
    a, aa = choose_action(Q, p_rand)
    return str(p_rand)+str(a)+str(aa)

map_path = sys.args[2]
map_img = cv2.imread(map_path)
map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
map_int = map_img.astype('int32')
# hyperparameters
dis = 50
dis_interpolate = 15
win_size = 30
color_low_bound = [60,120,120]
color_high_bound = [180,225,225]
# feature_color = [200,255,255]
for root, dirs, files in os.walk(sys.args[1]):
    for f in files:
        if not os.path.exists(sys.args[3]+'\\'+f[:-4]+"_align.txt"):
            inputfile = open(root+'/'+f, 'r')
            outputfile = open(sys.args[3]+'\\'+f[:-4]+"_align.txt", 'w')
            outputfile.writelines('x,y'+'\n')
            tmpt = []
            points = []
            print f
            for line in inputfile:
                q = line.strip().split(",")
                tmpt.append([int(q[0]),int(q[1])])
                if len(tmpt) > 1:
                    if math.sqrt((tmpt[-1][0]-tmpt[0][0])**2+(tmpt[-1][1]-tmpt[0][1])**2) > dis:
                        for i in range(len(tmpt)-1):
                            points.extend(interpolation(tmpt[i], tmpt[i+1], dis_interpolate))
                        tmpt = [tmpt[-1]]
                        Q = []
                        movement = []
                        state_action_count = []
                        ppoints = []
                        for p in points:
                            sub_img = map_img[p[0]-win_size/2: p[0]+win_size/2, p[1]-win_size/2:p[1]+win_size/2]
                            foreground = grabCut(sub_img, win_size, color_low_bound, color_high_bound)
                            Q_individual, move_individual = Q_value_initial(foreground, win_size)
                            if Q_individual != [[-100]]*9:
                                ppoints.append(p)
                                Q.append(Q_individual)
                                movement.append(move_individual)
                        print points
                        print ppoints
                        print Q
                        print movement
                        # Sarse control process
                        if len(ppoints) > 2:
                            alpha = 0.9
                            decay_rate = 0.9
                            count = 0
                            flag = 0
                            QQ = {}
                            state = "00"*len(ppoints)
                            chosen_p = 0
                            a, aa = choose_action(Q, 0)
                            action = str(chosen_p)+str(a)+str(aa)
                            reward_record = [0.0]
                            max_reward = -1
                            max_action = "00"*len(ppoints)
                            while flag == 0 and count<10000:
                                points_move = copy.deepcopy(ppoints)
                                chosen_p = int(action[:-2])
                                a = int(action[-2])
                                aa = int(action[-1])
                                state_next = state[:2*chosen_p]+str(a)+str(aa)+state[2*chosen_p+2:]
                                for p in range(len(ppoints)):
                                    a = int(state_next[2*p])
                                    aa = int(state_next[2*p+1])
                                    if a!=0:
                                        points_move[p] = [points_move[p][0]+movement[p][a][aa][0], points_move[p][1]+movement[p][a][aa][1]]
                                reward = entire_line(map_int, points_move, color_low_bound, color_high_bound)
                                reward_record.append(reward)
                                print count, reward, state_next
                                pool = ThreadPoolExecutor(max_workers=8)
                                action_next = pool.submit(greedy, QQ, state_next, action, Q, ppoints, reward_record, movement, color_low_bound, color_high_bound).result() # tuple of args for foo
                                print action_next
                                action = action_next
                                state = state_next
                                count += 1
                                if reward>max_reward:
                                    max_action = state+action
                                    max_reward = reward
                                if 1 == reward:
                                    flag = 1
                                    max_action = state+action
                            if max_reward > 0.5:
                                for p in range(len(ppoints)):
                                    a = int(max_action[2*p])
                                    aa = int(max_action[2*p+1])
                                    if a != 0:
                                        res = [ppoints[p][0]+movement[p][a][aa][0], ppoints[p][1]+movement[p][a][aa][1]]
                                    else:
                                        res = ppoints[p]
                                    outputfile.writelines(str(res[1])+","+str(-res[0])+"\n")

                        points = []
