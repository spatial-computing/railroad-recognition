import numpy as np
import cv2

def buffer(pred_ske, buffer_size=1):
    if np.max(pred_ske)==0:
        print('blank img')
        return
    if np.max(pred_ske)!=1:
        pred_ske= pred_ske/np.max(pred_ske)

    for i in range(buffer_size):
        nonzeros_idx = np.where(pred_ske!=0)
        xs, ys = nonzeros_idx
        # west
        d1 = (xs-1, ys)
        # east
        d2 = (xs+1, ys)
        # north
        d3 = (xs, ys-1)
        # south
        d4 = (xs, ys+1)
        # northwest
        d5 = (xs-1, ys-1)
        # northeast
        d6 = (xs+1, ys-1)
        # southwest
        d7 = (xs-1, ys+1)
        # southeast
        d8 = (xs+1, ys+1)
        pred_ske[d1]=1
        pred_ske[d2]=1
        pred_ske[d3]=1
        pred_ske[d4]=1
        pred_ske[d5]=1
        pred_ske[d6]=1
        pred_ske[d7]=1
        pred_ske[d8]=1
    return pred_ske
