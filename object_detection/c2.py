# q(z|x,y) a mixture gaussian to relief the anti-cluster caused by prior of Y

from keras.layers import Lambda, Input, Dense, Merge, Concatenate,Multiply, Add, add, Activation
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam, SGD, Adagrad
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

def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return 0
        
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sampling_xy(args):
    z_mean_prior, z_log_var_prior, z = args
    zy = z_mean_prior + K.exp(0.5*z_log_var_prior) * z
    return zy

def log_normal(x, mu, log_var, eps=0.0, axis=-1):
    if eps > 0.0:
        log_var = tf.add(log_var, eps, name='clipped_var')
    return -0.5 * K.sum(K.log(2 * np.pi) + log_var + K.square(x - mu) / K.exp(log_var), axis=-1)
#     return -0.5 * K.sum(log_var + K.square(x - mu) / K.exp(log_var), axis=-1)
    

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# TARGET_SAMPLE_DIR = "./data/orcutt/target_samples"
# MAP_PATH = './data/orcutt/CA_Orcutt_293752_1959_24000_geo.png'
# MASK_PATH = './data/orcutt/CA_Orcutt_293752_1959_24000_wetlands.png'
# TARGET_SAMPLE_DIR = "./data/bray_2001/target_samples"
# MAP_PATH = './data/bray_2001/CA_Bray_100414_2001_24000_geo.png'
# MASK_PATH = './data/bray_2001/wetlands.png'
# SUBSET_PATH = './data/bray_2001/subset'
# TARGET_SAMPLE_DIR = "./data/big_swamp/target_samples"
# MAP_PATH = './data/big_swamp/CA_Big Swamp_100195_1993_24000_geo.png'
# MASK_PATH = './data/big_swamp/wetlands.png'
TARGET_SAMPLE_DIR = "./data/car/target_samples"
MAP_PATH = "./data/car/Toronto_03553.png"
MASK_PATH = "./data/car/parking_mask3.png"
SUBSET_PATH = './data/car/subset3'
# TARGET_SAMPLE_DIR = "./data/car_sacramento/target_samples"
# MAP_PATH = "./data/car_sacramento/scaramento_parking.jpg"
# MASK_PATH = "./data/car_sacramento/mask_car_sacramento.png"
# SUBSET_PATH = './data/plan/subset'
# TARGET_SAMPLE_DIR = "./data/plan/target_samples"
# MAP_PATH = "./data/plan/plan.jpg"
# MASK_PATH = "./data/plan/plan_mask.png"
SHIFT_LIST = [0,5]#[-20,-15,-10,-5,0,5,10,15,20] #[-10,-5,0,5,10]#[0,2][0,3,6]#
ROTATION_ANGLE = []#0,5,10,15,20,340,345,355
for i in range(0, 360, 90):
    ROTATION_ANGLE.append(i)
IMG_SIZE = 40
STRIDE = 10
LATENT_DIM = 2
TRAINING_PROPORTION = 0.25
REPRESENTATION_DIM = 20
REPRESENTATION_ACTIVATION = 'tanh'
EPOCHS = 1000
LEARNING_RATE = 0.0001
NUM_ITERATIONS = 1
VAE_MODEL_PATH = ''
LOG_DIR = './logs'
MODEL_PATH = ''
batch_size = 81
wetlands = [145,147,149,162,169,178,372,385]
others = [4,29,52,96,116,297,365,376]
latent_dim = 32
intermediate_dim = 1024
num_cls = 2
optimizer = Adam(lr=LEARNING_RATE)
initializer = 'glorot_normal'#'random_uniform'

# optimizer = SGD(lr=LEARNING_RATE)
# optimizer = Adagrad(lr=LEARNING_RATE)


####################################################################################
# load data
x_u, _ = load_data.load_all_data(MAP_PATH, MASK_PATH, IMG_SIZE, STRIDE)
# x_u, _ = load_data.load_wetland_samples(SUBSET_PATH)
np.random.shuffle(x_u)
x_l, target_name = load_data.load_wetland_samples(TARGET_SAMPLE_DIR)
# x_l = x_u[39:40]
# x_l_aug = data_augmentation.data_aug(x_l, SHIFT_LIST, ROTATION_ANGLE)
x_l_aug = x_l
# x_u = data_augmentation.data_aug(x_l, SHIFT_LIST, rotation)
print(x_l_aug.shape)

np.random.shuffle(x_l_aug)

x_l = np.reshape(x_l, [-1, IMG_SIZE*IMG_SIZE*3])
x_l_aug = np.reshape(x_l_aug, [-1, IMG_SIZE*IMG_SIZE*3])
x_u = np.reshape(x_u, [-1, IMG_SIZE*IMG_SIZE*3])
image_size = x_u.shape[1]
original_dim = image_size

x_u = x_u.astype('float32') / 255
x_l = x_l.astype('float32') / 255
x_l_aug = x_l_aug.astype('float32') / 255

np.random.shuffle(x_l_aug)
x_l_aug = np.vstack((x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug, x_l_aug,\
                    x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,\
                     x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,\
                    x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug,x_l_aug))
# x_l_aug0 = np.repeat(x_l_aug[np.newaxis,0,:], 36, axis=0)


print('target samples shape: ', x_l_aug.shape)
print('all samples shape: ', x_u.shape)
# print('target aug samples shape: ', x_l_aug0.shape,x_l_aug1.shape,x_l_aug2.shape,x_l_aug3.shape)
####################################################################################

############################################################################
# p(z|c)
c_inputs0 = Input(shape=(num_cls,), name='c_encoder_inputs0')
c_inputs1 = Input(shape=(num_cls,), name='c_encoder_inputs1')

c_x = Dense(128, activation='relu',kernel_initializer = initializer)
c_xx = Dense(256, activation='relu',kernel_initializer = initializer)
# c_xxx = Dense(256, activation='relu')
c_mean = Dense(latent_dim, name='c_z_mean0',kernel_initializer = initializer)
c_var = Dense(latent_dim, name='c_z_log_var0',kernel_initializer = initializer)

c_x1 = Dense(128, activation='relu',kernel_initializer = initializer)
c_xx1 = Dense(256, activation='relu',kernel_initializer = initializer)
# c_xxx1 = Dense(128, activation='relu')
c_mean1 = Dense(latent_dim, name='c_z_mean1',kernel_initializer = initializer)
c_var1 = Dense(latent_dim, name='c_z_log_var1',kernel_initializer = initializer)

c_x0 = c_x(c_inputs0)
c_x0 = c_xx(c_x0)
# c_x0 = c_xxx(c_x0)
c_z_mean0 = c_mean(c_x0)
c_z_log_var0 = c_var(c_x0)
c_z0 = Lambda(sampling, output_shape=(latent_dim,), name='c_z0')([c_z_mean0, c_z_log_var0])

c_x1 = c_x(c_inputs1)
c_x1 = c_xx(c_x1)
# c_x1 = c_xxx(c_x1)
c_z_mean1 = c_mean(c_x1)
c_z_log_var1 = c_var(c_x1)
c_z1 = Lambda(sampling, output_shape=(latent_dim,), name='c_z1')([c_z_mean1, c_z_log_var1])

c_encoder0 = Model(c_inputs0, [c_z_mean0, c_z_log_var0, c_z0], name='c_encoder0')
c_encoder1 = Model(c_inputs1, [c_z_mean1, c_z_log_var1, c_z1], name='c_encoder1')

#######################
############################################################################


############################################################################
# vae for unlabeled samples
############################
# encoder
u_inputs = Input(shape=(original_dim,), name='u_encoder_inputs')
l_inputs = Input(shape=(original_dim,), name='l_encoder_inputs')

# p(y|x)
cls_x = Dense(256, activation='relu',kernel_initializer = 'random_uniform')(u_inputs)
cls_x = Dense(128, activation='relu',kernel_initializer = 'random_uniform')(cls_x)
u_qy_logit = Dense(2,kernel_initializer = 'random_uniform')(cls_x)
# cls = Model(u_inputs, cls_x)
u_qy = Activation('softmax')(u_qy_logit)
# l_qy = cls(l_inputs)

concat = Concatenate(axis=-1)
u_enc_x = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
u_enc_xx = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
u_mean = Dense(latent_dim, name='u_z_mean',kernel_initializer = initializer)
u_var = Dense(latent_dim, name='u_z_log_var',kernel_initializer = initializer)

l_enc_x = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
l_enc_xx = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
l_mean = Dense(latent_dim, name='l_z_mean',kernel_initializer = initializer)
l_var = Dense(latent_dim, name='l_z_log_var',kernel_initializer = initializer)

u0_concat = concat([u_inputs, c_inputs0])
u1_concat = concat([u_inputs, c_inputs1])
l_concat = concat([l_inputs, c_inputs1])

u_x0 = u_enc_x(u0_concat)
u_x0 = u_enc_xx(u_x0)
u_z_mean0 = u_mean(u_x0)
u_z_log_var0 = u_var(u_x0)

u_x1 = l_enc_x(u1_concat)
u_x1 = l_enc_xx(u_x1)
u_z_mean1 = l_mean(u_x1)
u_z_log_var1 = l_var(u_x1)

l_x = l_enc_x(l_concat)
l_x = l_enc_xx(l_x)
l_z_mean = l_mean(l_x)
l_z_log_var = l_var(l_x)

u_z1 = Lambda(sampling, output_shape=(latent_dim,), name='u_z1')([u_z_mean1, u_z_log_var1])
u_z0 = Lambda(sampling, output_shape=(latent_dim,), name='u_z0')([u_z_mean0, u_z_log_var0])
l_z = Lambda(sampling, output_shape=(latent_dim,), name='l_z')([l_z_mean, l_z_log_var])

# p(y|z)
# cls_x = Dense(64, activation='relu',kernel_initializer='random_uniform')(u_z)
# u_qy= Dense(num_cls, activation='softmax',kernel_initializer='random_uniform')(cls_x)

# instantiate encoder model
u_encoder0 = Model([u_inputs,c_inputs0], [u_z_mean0, u_z_log_var0, u_z0], name='u_encoder0')
u_encoder1 = Model([u_inputs,c_inputs1], [u_z_mean1, u_z_log_var1, u_z1], name='u_encoder1')
l_encoder = Model([l_inputs,c_inputs1], [l_z_mean, l_z_log_var, l_z], name='l_encoder')
# u_encoder = Model(u_inputs, [u_z_mean, u_z_log_var, u_z], name='u_encoder')
############################

############################
# decoder
u_latent_inputs0 = Input(shape=(latent_dim,), name='u_z_sampling0')
u_latent_inputs1 = Input(shape=(latent_dim,), name='u_z_samplin1')
l_latent_inputs = Input(shape=(latent_dim,), name='l_z_sampling')

u_dec_x = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
u_dec_xx = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
u_dec_outputs = Dense(original_dim, activation='sigmoid',kernel_initializer = initializer)

l_dec_x = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
l_dec_xx = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)
l_dec_outputs = Dense(original_dim, activation='sigmoid',kernel_initializer = initializer)


u_x0 = u_dec_x(u_latent_inputs0)
u_x0 = u_dec_xx(u_x0)
u_recons0 = u_dec_outputs(u_x0)

u_x1 = l_dec_x(u_latent_inputs1)
u_x1 = l_dec_xx(u_x1)
u_recons1 = l_dec_outputs(u_x1)

l_x = l_dec_x(l_latent_inputs)
l_x = l_dec_xx(l_x)
l_recons = l_dec_outputs(l_x)

u_decoder0 = Model(u_latent_inputs0, u_recons0, name='u_decoder0')
u_decoder1 = Model(u_latent_inputs1, u_recons1, name='u_decoder1')
l_decoder = Model(l_latent_inputs, l_recons, name='l_decoder')
# u_decoder.summary()
# u_outputs = u_decoder(u_z)
l_outputs = l_decoder(l_z)
u_outputs0 = u_decoder0(u_z0)
u_outputs1 = u_decoder1(u_z1)


############################################################################
# losses for vae
# compile and train vae 
u_reconstruction_loss0 = mse(u_inputs, u_outputs0)
u_reconstruction_loss0 *= original_dim  #* u_qy[:,0]
u_reconstruction_loss1 = mse(u_inputs, u_outputs1)
u_reconstruction_loss1 *= original_dim  #*  u_qy[:,1]
l_reconstruction_loss = mse(l_inputs, l_outputs)
l_reconstruction_loss *= original_dim

u_kl_cat = u_qy * K.log(u_qy / K.constant(np.array([0.5,0.5])))
u_kl_cat = K.sum(u_kl_cat, axis=-1) * 5.0


kl_loss_0 = (c_z_log_var0-u_z_log_var0)*0.5 + (K.square(u_z_mean0-c_z_mean0) + K.exp(u_z_log_var0)) / 2*K.exp(c_z_log_var0) - 0.5
kl_loss_0 = K.sum(kl_loss_0, axis=-1)


kl_loss_1 = (c_z_log_var1 - u_z_log_var1)*0.5 + (K.square(u_z_mean1-c_z_mean1) + K.exp(u_z_log_var1)) / 2*K.exp(c_z_log_var1) - 0.5
kl_loss_1 = K.sum(kl_loss_1, axis=-1)

l_kl_loss_1 = (c_z_log_var1 - l_z_log_var)*0.5 + (K.square(l_z_mean-c_z_mean1) + K.exp(l_z_log_var)) / 2*K.exp(c_z_log_var1) - 0.5
l_kl_loss_1 = K.sum(l_kl_loss_1, axis=-1) 


u_vae_loss = K.mean(u_qy[:,0]*(u_reconstruction_loss0+ kl_loss_0) +\
                    u_qy[:,1]*(u_reconstruction_loss1+ kl_loss_1) + l_reconstruction_loss+l_kl_loss_1+u_kl_cat)


u_vae = Model([u_inputs, l_inputs,c_inputs0,c_inputs1], [u_outputs0,u_outputs1,l_outputs,u_qy])
u_vae.add_loss(u_vae_loss)
u_vae.summary()
u_vae.compile(optimizer=optimizer, loss=None)

checkpoint = ModelCheckpoint('./logs/weights{epoch:08d}.h5',save_weights_only=True, period=100)


u_vae.load_weights('unlabel_plan.hdf5')


u_vae.fit([x_u[:136],x_l_aug[:136],np.array([[1,0]]*136),np.array([[0,1]]*136)], \
          epochs=200, batch_size=136, verbose=1, callbacks=[checkpoint])

############################################################################

############################################################################

x_u_pred0,x_u_pred1,x_l_pred,y_u_pred = u_vae.predict([x_u[:],x_l_aug[:x_u.shape[0]],np.array([[1,0]]*x_u.shape[0])\
           ,np.array([[0,1]]*x_u.shape[0])])

print(y_u_pred)
remove_files('./check_data/filtered')
remove_files('./check_data/all')
remove_files('./check_data/others')

for i in range(x_u_pred0.shape[0]): 
    if y_u_pred[i, 1] > 0.5:
        print(i, y_u_pred[i])
        cv2.imwrite('./check_data/filtered/'+str(i)+'_orig.png', cv2.cvtColor(x_u[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'_orig.png', cv2.cvtColor(x_u[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/all/'+str(i)+'.png', cv2.cvtColor(x_u_pred0[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/others/'+str(i)+'_orig.png', cv2.cvtColor(x_u[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./check_data/others/'+str(i)+'.png', cv2.cvtColor(x_u_pred1[i].reshape((IMG_SIZE, IMG_SIZE, 3))*255, cv2.COLOR_RGB2BGR))
