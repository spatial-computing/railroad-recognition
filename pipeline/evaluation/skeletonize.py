from skimage.morphology import skeletonize
import numpy as np
import cv2
import os

def preprocess(img):
    img = (img == 255).astype(np.bool)
    return img

def make_skeleton(root, fix_borders=True, debug=False):
    replicate = 5
    clip = 2
    rec = replicate + clip
    # open and skeletonize
    img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
    print(img.shape)

    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
    img = preprocess(img)
    if not np.any(img):
        return None, None
    ske = skeletonize(img).astype(np.uint16)
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
    return img_copy, ske
