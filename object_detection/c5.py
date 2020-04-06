# 5 clusters

from keras.layers import Lambda, Input, Dense, Merge, Concatenate,Multiply, Add, add, Activation
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import metrics

from utils import load_data, data_augmentation
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from models import vae, categorical
from keras import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras.datasets import mnist

def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return 0
        
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def log_normal(x, mu, log_var, eps=0.0, axis=-1):
    if eps > 0.0:
        log_var = tf.add(log_var, eps, name='clipped_var')
    return -0.5 * K.sum(K.log(2 * np.pi) + log_var + K.square(x - mu) / K.exp(log_var), axis=-1)
#     return -0.5 * K.sum(log_var + K.square(x - mu) / K.exp(log_var), axis=-1)

class MyCallback(keras.callbacks.Callback):
    def __init__(self, weight):
        self.weight = weight
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.weight) <= 1.0:
            self.weight = K.variable(1.0)
        else:
            self.weight = self.weight * 1.0/(1.0+1e-5*epoch)
        print ("Current ce Weight is " + str(K.get_value(self.weight)))

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# TARGET_SAMPLE_DIR = "./data/orcutt/target_samples_v1"
# MAP_PATH = './data/orcutt/CA_Orcutt_293752_1959_24000_geo.png'
# MASK_PATH = './data/orcutt/CA_Orcutt_293752_1959_24000_wetlands.png'
# SUBSET_PATH = './data/orcutt/subset'
TARGET_SAMPLE_DIR = "./data/bray_2001/target_samples"
MAP_PATH = './data/bray_2001/CA_Bray_100414_2001_24000_geo.png'
MASK_PATH = './data/bray_2001/wetlands.png'
SUBSET_PATH = './data/bray_2001/subset_v3'
# TARGET_SAMPLE_DIR = "./data/big_swamp/target_samples"
# MAP_PATH = './data/big_swamp/CA_Big Swamp_100195_1993_24000_geo.png'
# MASK_PATH = './data/big_swamp/wetlands.png'
# TARGET_SAMPLE_DIR = "./data/car/target_samples"
# MAP_PATH = "./data/car/Toronto_03553.png"
# MASK_PATH = "./data/car/parking_mask3.png"
# SUBSET_PATH = './data/car/subset3'
# TARGET_SAMPLE_DIR = "./data/car_sacramento/target_samples"
# MAP_PATH = "./data/car_sacramento/scaramento_parking.jpg"
# MASK_PATH = "./data/car_sacramento/mask_car_sacramento.png"
# SUBSET_PATH = './data/plan/subset'
# TARGET_SAMPLE_DIR = "./data/plan/target_samples"
# MAP_PATH = "./data/plan/plan.jpg"
# MASK_PATH = "./data/plan/plan_mask.png"
SHIFT_LIST = [-2,0,2] #
ROTATION_ANGLE = []
# for i in range(0, 360, 90):
#     ROTATION_ANGLE.append(i)
IMG_SIZE = 80
STRIDE = 30
EPOCHS = 1000
LEARNING_RATE = 0.0001
VAE_MODEL_PATH = ''
LOG_DIR = './logs'
MODEL_PATH = ''
latent_dim = 128#32, 5
intermediate_dim = 512#128, 512
num_cls = 5
optimizer = Adam(lr=LEARNING_RATE)
# optimizer = RMSprop(lr=LEARNING_RATE)
initializer = 'glorot_normal'#'random_uniform'#
original_dim = IMG_SIZE*IMG_SIZE*3
w_recons, w_kl, w_ce = 80.0*80.0, 1.0, 2000.0

def qz_graph(x, y, intermediate_dim=512,latent_dim=32):
    concat = Concatenate(axis=-1)([x, y])
    layer1 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(concat)
    layer2 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(layer1)
#     concat = Concatenate(axis=-1)([layer2, y])
    z_mean = Dense(latent_dim,kernel_initializer = initializer)(layer2)
    z_var = Dense(latent_dim,kernel_initializer = initializer)(layer2)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
    return z_mean, z_var, z

def qy_graph(x, num_cls=5):
    layer1 = Dense(256, activation='relu',kernel_initializer = initializer)(x)#256. 64
    layer2 = Dense(128, activation='relu',kernel_initializer = initializer)(layer1)#128, 32
    qy_logit = Dense(num_cls,kernel_initializer = initializer)(layer2)
    qy = Activation('softmax')(qy_logit)
    return qy_logit, qy

def px_graph(z, intermediate_dim=512, original_dim=40*40*3):
    layer1 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(z)
    layer2 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(layer1)
    reconstruction = Dense(original_dim, activation='sigmoid',kernel_initializer = initializer)(layer2)
    return reconstruction

def pzy_graph(y, latent_dim=32):
    h = Dense(16, activation='relu',kernel_initializer = initializer)(y)#128
    h = Dense(8, activation='relu',kernel_initializer = initializer)(h)#256, 64
    zp_mean = Dense(latent_dim,kernel_initializer = initializer)(h)
    zp_var = Dense(latent_dim,kernel_initializer = initializer)(h)
    return zp_mean, zp_var

def loss(x, xp, zm, zv, zm_prior, zv_prior, w_mse, w_kl):
    reconstruction_loss = mse(x, xp)
    reconstruction_loss *= w_mse
    kl_loss = (zv_prior-zv)*0.5 + (K.square(zm-zm_prior) + K.exp(zv)) / 2*K.exp(zv_prior) - 0.5
    kl_loss = K.sum(kl_loss, axis=-1) * w_kl
    return reconstruction_loss + kl_loss

def kl_loss(zm, zv, zm_prior, zv_prior, weight):
    loss = (zv_prior-zv)*0.5 + (np.square(zm-zm_prior) + np.exp(zv)) / 2*np.exp(zv_prior) - 0.5
    loss = np.sum(loss, axis=-1) * weight
    return loss

def mse_loss(x, xp, weight):
    return (np.square(x - xp)).mean(axis=None) * weight

def ce_loss(yp, weight):
    return (yp * np.log(yp / np.array([0.20,0.20,0.20,0.20,0.20]))).mean(axis=None) * weight

x_u = Input(shape=(original_dim,), name='x_u')
x_l = Input(shape=(original_dim,), name='x_l')
y0 = Input(shape=(num_cls,), name='y0_inputs')
y1 = Input(shape=(num_cls,), name='y1_inputs')
y2 = Input(shape=(num_cls,), name='y2_inputs')
y3 = Input(shape=(num_cls,), name='y3_inputs')
y4 = Input(shape=(num_cls,), name='y4_inputs')

zm_p0,zv_p0 = pzy_graph(y0, latent_dim=latent_dim)
zm_p1,zv_p1 = pzy_graph(y1, latent_dim=latent_dim)
zm_p2,zv_p2 = pzy_graph(y2, latent_dim=latent_dim)
zm_p3,zv_p3 = pzy_graph(y3, latent_dim=latent_dim)
zm_p4,zv_p4 = pzy_graph(y4, latent_dim=latent_dim)

zm, zv, z = qz_graph(x_l, y0, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
zm0, zv0, z0 = qz_graph(x_u, y0, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
zm1, zv1, z1 = qz_graph(x_u, y1, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
zm2, zv2, z2 = qz_graph(x_u, y2, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
zm3, zv3, z3 = qz_graph(x_u, y3, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
zm4, zv4, z4 = qz_graph(x_u, y4, intermediate_dim=intermediate_dim, latent_dim=latent_dim)

xp_l = px_graph(z, intermediate_dim=intermediate_dim, original_dim=original_dim)
xp_u0 = px_graph(z0, intermediate_dim=intermediate_dim, original_dim=original_dim)
xp_u1 = px_graph(z1, intermediate_dim=intermediate_dim, original_dim=original_dim)
xp_u2 = px_graph(z2, intermediate_dim=intermediate_dim, original_dim=original_dim)
xp_u3 = px_graph(z3, intermediate_dim=intermediate_dim, original_dim=original_dim)
xp_u4 = px_graph(z4, intermediate_dim=intermediate_dim, original_dim=original_dim)

qy_logit, qy = qy_graph(x_u)


vae = Model([x_l,x_u,y0,y1,y2,y3,y4], [xp_l,xp_u0,xp_u1,xp_u2,xp_u3,xp_u4,qy,\
                                   zm,zv,zm0,zv0,zm1,zv1,zm2,zv2,zm3,zv3,zm4,zv4,\
                                   zm_p0,zv_p0,zm_p1,zv_p1,zm_p2,zv_p2,zm_p3,zv_p3,zm_p4,zv_p4])

cat_loss = qy * K.log(qy / K.constant(np.array([0.20,0.20,0.20,0.20,0.20])))
# cat_loss = -1.0 * tf.keras.backend.categorical_crossentropy(qy_logit, qy, from_logits=False)
cat_loss = K.sum(cat_loss, axis=-1) * w_ce

vae_loss = qy[:,0]*loss(x_u,xp_u0,zm0,zv0,zm_p0,zv_p0,w_recons,w_kl)+\
            qy[:,1]*loss(x_u,xp_u1,zm1,zv1,zm_p1,zv_p1,w_recons,w_kl)+\
            qy[:,2]*loss(x_u,xp_u2,zm2,zv2,zm_p2,zv_p2,w_recons,w_kl) + \
            qy[:,3]*loss(x_u,xp_u3,zm3,zv3,zm_p3,zv_p3,w_recons,w_kl) + \
            qy[:,4]*loss(x_u,xp_u4,zm4,zv4,zm_p4,zv_p4,w_recons,w_kl) + cat_loss +\
            loss(x_l, xp_l, zm, zv, zm_p0, zv_p0, w_recons, w_kl)

vae.add_loss(vae_loss)

vae.summary()

x_u, img_names = load_data.load_wetland_samples(SUBSET_PATH)
_x, target_name = load_data.load_wetland_samples(TARGET_SAMPLE_DIR)
_x_aug = data_augmentation.data_aug(_x, SHIFT_LIST, ROTATION_ANGLE)
_x_aug = np.vstack((_x_aug, _x_aug,_x_aug))
# x_u, img_names = load_data.load_all_data(MAP_PATH, MASK_PATH, IMG_SIZE, STRIDE)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# train_filter = np.where((y_train == 0 ) | (y_train == 1) | (y_train == 2))
# x_u = x_train[train_filter]
np.random.shuffle(_x_aug)
x_u = np.reshape(x_u, [-1, IMG_SIZE*IMG_SIZE*3])
_x_aug = np.reshape(_x_aug, [-1, IMG_SIZE*IMG_SIZE*3])
x_u = x_u.astype('float32') / 255
_x_aug = _x_aug.astype('float32') / 255
print(x_u.shape, _x_aug.shape)

vae.compile(optimizer=optimizer, loss=None)
batch_size = x_u.shape[0]
_x_aug = _x_aug[:batch_size]
checkpoint = ModelCheckpoint('./logs/weights{epoch:08d}.h5',save_weights_only=True, period=100)
# vae.load_weights('./logs/weights00000100.h5')
vae.load_weights('vae3_i2_.hdf5')
vae.fit([_x_aug,x_u,np.array([[1,0,0,0,0]]*batch_size),np.array([[0,1,0,0,0]]*batch_size),np.array([[0,0,1,0,0]]*batch_size), \
              np.array([[0,0,0,1,0]]*batch_size),np.array([[0,0,0,0,1]]*batch_size)],epochs=400, batch_size=batch_size, verbose=1, callbacks=[checkpoint])
vae.save_weights('vae3_i3.hdf5')

x_l_rec,x_c0, x_c1, x_c2, x_c3, x_c4, y_pred,m,v,m0,v0,m1,v1,m2,v2,m3,v3,m4,v4,mp0,vp0,mp1,vp1,mp2,vp2,mp3,vp3,mp4,vp4 =\
vae.predict([_x_aug,x_u,np.array([[1,0,0,0,0]]*batch_size),np.array([[0,1,0,0,0]]*batch_size),np.array([[0,0,1,0,0]]*batch_size), \
              np.array([[0,0,0,1,0]]*batch_size),np.array([[0,0,0,0,1]]*batch_size)])
print(y_pred)

# print('*************************************')    
# for i in range(y_pred.shape[0]):
#     print(m0[i], v2[i])
#     print(mp0[i], vp0[i])
#     print(i,np.argmax(y_pred[i]), y_pred[i])
#     print("kl loss: ", kl_loss(m0[i],v0[i],mp0[i],vp0[i],w_kl), kl_loss(m1[i],v1[i],mp1[i],vp1[i],w_kl), \
#           kl_loss(m2[i],v2[i],mp2[i],vp2[i],w_kl),kl_loss(m3[i],v3[i],mp3[i],vp3[i],w_kl)), \
#            kl_loss(m4[i],v4[i],mp4[i],vp4[i],w_kl)
#     print("mse loss: ", mse_loss(x_u[i],x_c0[i],w_recons),mse_loss(x_u[i],x_c1[i],w_recons),\
#           mse_loss(x_u[i],x_c2[i],w_recons),mse_loss(x_u[i],x_c3[i],w_recons),mse_loss(x_u[i],x_c4[i],w_recons))
#     print("ce loss: ", ce_loss(y_pred[i],w_ce))
#     print('******')

remove_files('./check_data/filtered')
remove_files('./check_data/all')
remove_files('./check_data/others')

for i in range(x_u.shape[0]): 
    if np.argmax(y_pred[i]) == 0:
        if img_names[i] not in target_name:
            cv2.imwrite(os.path.join('./check_data/filtered',img_names[i]), cv2.cvtColor(x_u[i].reshape((IMG_SIZE, IMG_SIZE,3))*255, cv2.COLOR_RGB2BGR))
#         cv2.imwrite(os.path.join('./check_data/filtered',str(i)+'_orig.png'), cv2.cvtColor(x_u[i].reshape((IMG_SIZE, IMG_SIZE,3))*255, cv2.COLOR_RGB2BGR))
#             print(img_names[i], y_u_pred[i])    
    cv2.imwrite('./check_data/all/'+str(i)+'_orig.png', cv2.cvtColor(x_u[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'_0.png', cv2.cvtColor(x_c0[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'_1.png', cv2.cvtColor(x_c1[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'_2.png', cv2.cvtColor(x_c2[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'_3.png', cv2.cvtColor(x_c3[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'_4.png', cv2.cvtColor(x_c4[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))

#     cv2.imwrite('./check_data/others/'+str(i)+'_orig.png', cv2.cvtColor(_x_aug[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
#     cv2.imwrite('./check_data/others/'+str(i)+'_0.png', cv2.cvtColor(x_l_rec[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
   
