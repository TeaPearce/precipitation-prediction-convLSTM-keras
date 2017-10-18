# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:05:33 2017

@author: Tim Pearce
"""

import pandas as pd
import numpy as np
import datetime
import os
import sys
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import h5py

from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling3D, LSTM, Reshape
from keras import backend as K
from keras.callbacks import History 

# -----------------------------------------------------------------------------
# misc functions
def fn_print(string):
    print("\n-- ", string, ": ",datetime.now().strftime('%H:%M:%S'),"--")


# -----------------------------------------------------------------------------
# read in data from txt and format
def fn_load_data(path_in, no_import, start_line):
    id_no = []
    label = []
    data = []
    
    # scan through file line by line
    with open(path_in) as infile:
        i=-1
        for line in infile:
            i+=1
            if i % 500 == 0:
                fn_print(("considering line:" + str(i)))
                #print("considering line:",i)
            if i < start_line:
                continue
            if i >= no_import + start_line:
                break
            temp = line.split(",")
            id_no.append(str(temp[0]))
            label.append(float(temp[1]))
            data_temp = temp[2].split(" ")
            # data_temp = [int(x) for x in data_temp] # prob slowest part
            data_temp = list(map(int, data_temp)) # slightly quicker
            data.append(data_temp)

    # save results
    np_id = np.array(id_no)
    np_label = np.array(label)
    np_data = np.array(data)
    
    # clear memory
    del id_no, label, data
    
    # reshape data
    np_data = np.reshape(np_data, newshape=-1)
    T,H,Y,X = 15,4,101,101
    np_data = np.reshape(np_data, newshape=(-1,T,H,Y,X), order='C')
    
    return np_id, np_label, np_data

# -----------------------------------------------------------------------------
# load data into .h5 file

def fn_h5_append(h5f_name, name_in, data_in):
    h5f = h5py.File(h5f_name, 'a')
    h5f.create_dataset(name_in, data=data_in)
    h5f.close()

# create h5 file
#h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_v01.h5'
#h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_testA_v02.h5'
#h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_testB_v01.h5'
#h5f = h5py.File(h5f_name, 'w')
#grp = h5f.create_group("test_B")
#h5f.close()
### commented out to avoid retriggering and overwriting again! ###

# iterate through txt file, and write to h5 file
step_size = 2000
#path_in = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\data_new\\CIKM2017_train\\train.txt"
#path_in = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\data_new\\CIKM2017_testA\\testA.txt"
path_in = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\data_new\\CIKM2017_testB\\testB.txt"
for i in np.arange(0,2000,step_size):
    
    fn_print(("convert to h5, outter loop:"+str(i)))
    np_train_id, np_train_label, np_train_data = fn_load_data(path_in, start_line=i, no_import=step_size)
    
    filename_id = ('/test_B/test_id_'+str(i)+"_to_"+str(i+step_size-1))
    filename_label = ('/test_B/test_label_'+str(i)+"_to_"+str(i+step_size-1))
    filename_data = ('/test_B/test_data_'+str(i)+"_to_"+str(i+step_size-1))
#    filename_id = ('/test_A/test_id_'+str(i)+"_to_"+str(i+step_size-1))
#    filename_label = ('/test_A/test_label_'+str(i)+"_to_"+str(i+step_size-1))
#    filename_data = ('/test_A/test_data_'+str(i)+"_to_"+str(i+step_size-1))
#    filename_id = ('/train/train_id_'+str(i)+"_to_"+str(i+step_size-1))
#    filename_label = ('/train/train_label_'+str(i)+"_to_"+str(i+step_size-1))
#    filename_data = ('/train/train_data_'+str(i)+"_to_"+str(i+step_size-1))
    
    # need to convert unicode -> ascii
    ascii_id = [n.encode("ascii", "ignore") for n in np_train_id.tolist()]
    
    fn_h5_append(h5f_name, filename_id, ascii_id)
    fn_h5_append(h5f_name, filename_label, np_train_label)
    fn_h5_append(h5f_name, filename_data, np_train_data)
    
    del np_train_id, np_train_label, np_train_data


                    #######################
                    ### start from here ###
                    #######################


# -----------------------------------------------------------------------------
# read datd from .h5 file
def fn_h5_to_np(test_train, i=0):
    # i is the start line (0,1,2...9) used for training
    # i is the start line (0,1,2...4) used for train_shuffle (folds 1 to 8)
    # i is the start line (0,1,2...4) train_shuffle_full (folds 1 to 10)
    
    if test_train == "testA":
        filename_id = ('/test_A/test_id_0_to_1999')
        filename_label = ('/test_A/test_label_0_to_1999')
        filename_data = ('/test_A/test_data_0_to_1999')
        h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_testA_v02.h5'
        
    elif test_train == "testB":
        filename_id = ('/test_B/test_id_0_to_1999')
        filename_label = ('/test_B/test_label_0_to_1999')
        filename_data = ('/test_B/test_data_0_to_1999')
        h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_testB_v01.h5'
        
    elif test_train == "train": # not shuffled - sequential - order 0 to 9
        step_size = 1000
        i*=1000
        filename_id = ('/train/train_id_'+str(i)+"_to_"+str(i+step_size-1))
        filename_label = ('/train/train_label_'+str(i)+"_to_"+str(i+step_size-1))
        filename_data = ('/train/train_data_'+str(i)+"_to_"+str(i+step_size-1))
        h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_v01.h5'
    
    elif test_train == "train_shuffle": # shuffled only train folds 0 to 7
        filename_id = ('/train/train_id_fold_'+str(i))
        filename_label = ('/train/train_label_fold_'+str(i))
        filename_data = ('/train/train_data_fold_'+str(i))
        h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_train_shuffled_1to8_v01.h5'
    
    elif test_train == "train_shuffle_full": # shuffled all train folds 0 to 9
        filename_id = ('/train/train_id_fold_'+str(i))
        filename_label = ('/train/train_label_fold_'+str(i))
        filename_data = ('/train/train_data_fold_'+str(i))
        h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_train_shuffled_1to10_v01.h5'
        
    h5f = h5py.File(h5f_name,'r')
    np_id = h5f[filename_id][:]
    np_label = h5f[filename_label][:]
    np_data = h5f[filename_data][:]
    h5f.close()
    
    return np_id, np_label, np_data


# -----------------------------------------------------------------------------
# read / shuffle / rewrite .h5 file data
# run training, by shuffling each fold (0 to 8) and picking chunks of samples from each

# create h5 file
#h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_v01.h5'
#h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_train_shuffled_1to8_v01.h5'
#h5f_name = 'D:\\02 Datasets\\05 Precipitation Nowcasting data\\tp_data\\data_train_shuffled_1to10_v01.h5'
#h5f = h5py.File(h5f_name, 'w')
#grp = h5f.create_group("train")
#h5f.close()

no_iter = 5
for fold in range(0,no_iter):
    fn_print(("training on fold:"+str(fold)))
    for i in range(0,10): # folds
        fn_print(("selecting data from group:"+str(i)))
        np_train_id, np_train_label, np_train_data  = fn_h5_to_np(test_train="train",i=i)
        
        np.random.seed(10) # want same permutation for each fold
        perm = np.random.permutation(np_train_data.shape[0])
        batch_size = int(np.floor(np_train_data.shape[0]/no_iter))
        
        if i == 0: # create array
            np_X_train = np_train_data[perm[batch_size*fold:batch_size*(fold+1)]]
            np_y_train = np_train_label[perm[batch_size*fold:batch_size*(fold+1)]]
            np_id_train = np_train_id[perm[batch_size*fold:batch_size*(fold+1)]]
        else: # append
            np_X_train = np.concatenate((np_X_train,np_train_data[perm[batch_size*fold:batch_size*(fold+1)]]),axis=0)
            np_y_train = np.concatenate((np_y_train,np_train_label[perm[batch_size*fold:batch_size*(fold+1)]]),axis=0)
            np_id_train = np.concatenate((np_id_train,np_train_id[perm[batch_size*fold:batch_size*(fold+1)]]),axis=0)
    
    # shuffle once more
    np.random.seed(10)
    perm = np.random.permutation(np_X_train.shape[0])
    np_X_train = np_X_train[perm]
    np_y_train = np_y_train[perm]
    np_id_train = np_id_train[perm]
    
    filename_id = ('/train/train_id_fold_'+str(fold))
    filename_label = ('/train/train_label_fold_'+str(fold))
    filename_data = ('/train/train_data_fold_'+str(fold))
    
    fn_h5_append(h5f_name, filename_id, np_id_train)
    fn_h5_append(h5f_name, filename_label, np_y_train)
    fn_h5_append(h5f_name, filename_data, np_X_train)


# -----------------------------------------------------------------------------
# load data from hd5
#np_train_id, np_train_label, np_train_data  = fn_h5_to_np(test_train="train",i=1)
#np_train_id, np_train_label, np_train_data  = fn_h5_to_np(test_train="train_shuffle",i=0)
#np_train_id, np_train_label, np_train_data  = fn_h5_to_np(test_train="train_shuffle_full",i=0)
#np_test_id, np_test_label, np_test_data = fn_h5_to_np(test_train="testA")

# -----------------------------------------------------------------------------
# visualise
# heatmap of one radar image at each height
s_view = 1501
for t_view in [0,5,9,14]:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,0], cmap='plasma')
    ax2.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,1], cmap='plasma')
    ax3.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,2], cmap='plasma')
    ax4.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,3], cmap='plasma')
    ax1.set_title("height_0")
print("rainfall",np_train_label[s_view])

s_view = 0
for t_view in [0,1,2,3]:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    ax1.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,0], cmap='plasma')
    ax2.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,1], cmap='plasma')
    ax3.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,2], cmap='plasma')
    ax4.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,t_view,3], cmap='plasma')
print("rainfall",np_train_label[s_view])

# observations about training sample:
# height 0 appears to have large blank areas e.g. s_view = 2
# s_view = 3, s_view = 2 seem to overlap (same shaped blank area)
# also a scattering of clumps of pixels are blank - height 0 again

# over time
s_view = 1
h_view = 0
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,0,h_view], cmap='plasma')
ax2.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,4,h_view], cmap='plasma')
ax3.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,9,h_view], cmap='plasma')
ax4.pcolormesh(np.arange(101), np.arange(101), np_train_data[s_view,14,h_view], cmap='plasma')

# need to clean to fill in missing pixels
# e.g. np_train_data[0,14,0]

# find interesting ones
np_label.argmax()
np_label.argmin()

# -----------------------------------------------------------------------------
# animation

from matplotlib import pyplot as plt
from matplotlib import animation

np_id, np_label, np_data  = fn_h5_to_np(test_train="train",i=8)

nx = np_data.shape[3]
ny = np_data.shape[4]

s_view = 54# 685 # 54
h_view = 2

fig = plt.figure()
data = np_data[s_view,:,h_view]
#im = plt.imshow(data[0,:], cmap='gist_gray_r', vmin=0, vmax=1)
im = plt.imshow(data[0,:], cmap='gist_gray_r', vmin=0, vmax=data[:].max())

def init():
    im.set_data(data[0,:])
    
def animate(i):
    im.set_data(data[i,:])
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nx * ny, interval=500)
print("rainfall",np_label[s_view])

# look at what mean pooling does


import tensorflow as tf
sess = tf.InteractiveSession()
temp = K.square(y_pred - y_true) # 1600, 1
temp = y_pred - y_true
temp = K.square(temp)
temp = K.square(4)

temp.eval()
out = temp.eval()
print(out)

sess.close()




from keras import backend as K
import tensorflow as tf

sess = tf.InteractiveSession()

data = np_data[s_view,0,h_view]
data = data.astype(float)
data = data.reshape(1,101,101,1)

tmp_tensor = K.pool2d(data, pool_size=(4,4), strides=(4, 4), padding='valid', pool_mode='max')
out = tmp_tensor.eval()

data_out = out.reshape(25,25)
data_in = data.reshape(101,101)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.pcolormesh(np.arange(101), np.arange(101), data_in, cmap='gist_gray_r')
ax2.pcolormesh(np.arange(25), np.arange(25), data_out, cmap='gist_gray_r')



# -----------------------------------------------------------------------------
# 'simple' model - use CNN with input of only one timeframe, one height (not zero)

# hardcoded these after brief analysis of training and testA
y_std = 15.
y_mean = 15.
X_std = 50.
X_mean = 60.

my_height = 3

# normalise input and output
def fn_norm_Xy(X,y,is_graph=False):
    # normalise X and y
    X = X / X_std
    #X = (X - X_mean) / X_std
    y = y / y_std # don't want y to be <0 as using relu
    if is_graph:
        fig, (ax1,ax2) = plt.subplots(2,1)
        ax1.hist(X.reshape(-1),bins=100)
        ax2.hist(y.reshape(-1),bins=100)
    return X, y

# deal with -1's in data
def fn_minus_ones(X):
    #(np_train_data < 0).sum()
    #np_train_data.reshape(-1).shape[0]
    
    ### TO DO ### something cleverer
    X[(X < 0)] = 0
    
    return X

# make binary
def fn_binary_clouds(X):
    X[(X >= 1)] = 1
    X[(X < 1)] = 0
    return X


# convert np arrays to X and y
def fn_np_to_Xy(np_data, np_label, h_select = 3, t_select = 14):
    
    X = np_data[:,t_select,h_select,:,:] # select all samples, all data
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    y = np_label
    y = y.reshape(y.shape[0], 1)
    
    return X, y

# from h5 file to Xy 2D, at one height
def fn_h5_to_Xy(test_train,i=0,h_select = 3, t_select = 14):
    np_id, np_label, np_data = fn_h5_to_np(test_train=test_train,i=i) # load data from h5
    X, y = fn_np_to_Xy(np_data, np_label, h_select = h_select, t_select = t_select) # convert to X train format
    X = fn_minus_ones(X)
    X, y = fn_norm_Xy(X,y,is_graph=False) # normalise
    return X,y


# convert np arrays to X and y - 2D + timeD, at one height
def fn_np_to_Xy_2D_timeD(np_data, np_label, h_select = 3):
    
    X = np_data[:,:,h_select,:,:] # select all samples, all data
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
    
    y = np_label
    y = y.reshape(y.shape[0], 1)
    
    return X, y

# from h5 file to Xy - 2D + timeD, at one height
def fn_h5_to_Xy_2D_timeD(test_train,i=0,h_select = 3):
    np_id, np_label, np_data = fn_h5_to_np(test_train=test_train,i=i) # load data from h5
    X, y = fn_np_to_Xy_2D_timeD(np_data, np_label, h_select = h_select) # convert to X,y format
    X = fn_minus_ones(X)
    
    X, y = fn_norm_Xy(X,y,is_graph=False) # normalise
    
    #X = fn_binary_clouds(X) # make binary
    return X,y



if False:
    # divide into test and train sets
    np.random.seed(10)
    perm = np.random.permutation(X.shape[0])
    train_size = int(np.floor(X.shape[0]*0.8))
    
    X_train = X[perm[:train_size]]
    X_val = X[perm[train_size:]]
    #X_test = X[perm[train_size:]] ### MUST ADJUST THIS LATER TO TAKE HALF OF THE STUFF ###
    
    y_train = y[perm[:train_size]]
    y_val = y[perm[train_size:]]
    #y_test = y[perm[train_size:]]


#def fn_keras_rmse(y_true, y_pred):
#    return K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std)), axis=-1))
#    #return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def fn_keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std))))

def fn_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def fn_get_model():

    # Model parameters
    rows, cols = 101, 101
    input_shape = (rows, cols, 1)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    #c1 = Convolution2D(filters=8, 7, 7, border_mode='same', init='he_uniform', activation='relu')(inpN)
    c1 = Conv2D(filters=8, kernel_size= (7,7), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(inpN)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters=8, kernel_size= (7,7), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling2D()(c1)
    #drop_1 = Dropout(0.25)(pool_1)
    
    c2 = Conv2D(filters=16, kernel_size= (5,5), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(pool_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters=16, kernel_size= (5,5), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling2D()(c2)
    #drop_2 = Dropout(0.25)(pool_2)
    
    c3 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(pool_2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    pool_3 = MaxPooling2D()(c3)
    drop_3 = Dropout(0.25)(pool_3)
    
    c4 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=32, kernel_size= (3,3), strides=(1, 1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    pool_4 = MaxPooling2D()(c4)
    drop_4 = Dropout(0.25)(pool_4)
    
    flat = Flatten()(drop_4)
    hidden_1 = Dense(1024, kernel_initializer='glorot_uniform', activation='relu')(flat)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.3)(hidden_2)
    hidden_3 = Dense(124, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.3)(hidden_3)
    out = Dense(1, activation='relu')(hidden_3)

    model = Model(outputs=out, inputs=inp)

    return model


def fn_get_model_2D_timeD():

    t_frames, rows, cols = 15, 101, 101
    input_shape = (t_frames, rows, cols, 1)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    
    c1 = Conv3D(filters=8, kernel_size= (5,5,5), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(inpN)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(filters=8, kernel_size= (5,5,5), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling3D(pool_size=(2,3,3))(c1)
    drop_1 = Dropout(0.25)(pool_1)
    
    c2 = Conv3D(filters=16, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(filters=16, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling3D(pool_size=(1,3,3))(c2)
    drop_1 = Dropout(0.25)(pool_2)
    
    c3 = Conv3D(filters=32, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_1)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(filters=32, kernel_size= (3,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    pool_3 = MaxPooling3D(pool_size=(1,2,2))(c3)
    drop_3 = Dropout(0.25)(pool_3)

    flat = Flatten()(drop_3)
    hidden_1 = Dense(1024, kernel_initializer='glorot_uniform', activation='relu')(flat)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.3)(hidden_2)
    hidden_3 = Dense(124, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.3)(hidden_3)
    hidden_4 = Dense(8, kernel_initializer='glorot_uniform', activation='relu')(hidden_3)
    hidden_4 = BatchNormalization()(hidden_4)
    hidden_4 = Dropout(0.3)(hidden_4)
    out = Dense(1, activation='relu')(hidden_4)
    
    model = Model(outputs=out, inputs=inp)
    
    print(model.summary())

    return model

def fn_get_model_2D_LSTM_1():

    t_frames, rows, cols = 15, 101, 101
    input_shape = (t_frames, rows, cols, 1)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    pool_1 = AveragePooling3D(pool_size=(2,4,4))(inpN) # downsample # 7,25,25,1

    #flat = Flatten()(pool_1)
    
    flat = Reshape((7,625))(pool_1)
    flat = BatchNormalization()(flat)
    flat = Dropout(0.1)(flat)
    
    lstm_1 = LSTM(units=625, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.2, recurrent_dropout=0.2)(flat)
    
    hidden_1 = Dense(625, kernel_initializer='glorot_uniform', activation='relu')(flat)
    hidden_1 = BatchNormalization()(lstm_1)
    hidden_1 = Dropout(0.3)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.3)(hidden_2)
    hidden_3 = Dense(124, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.3)(hidden_3)
    hidden_4 = Dense(8, kernel_initializer='glorot_uniform', activation='relu')(hidden_3)
    hidden_4 = BatchNormalization()(hidden_4)
    hidden_4 = Dropout(0.3)(hidden_4)
    out = Dense(1, activation='linear')(hidden_4)
    
    model = Model(outputs=out, inputs=inp)
    
    print(model.summary())

    return model


def fn_get_model_2D_conv_to_LSTM_1():

    t_frames, rows, cols = 15, 101, 101
    input_shape = (t_frames, rows, cols, 1)

    inp = Input(shape=input_shape)
    inpN = BatchNormalization()(inp)
    pool_0 = AveragePooling3D(pool_size=(1,4,4))(inpN) # downsample slightly ???
    
    c1 = Conv3D(filters=8, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(pool_0)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(filters=8, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    pool_1 = MaxPooling3D(pool_size=(1,2,2))(c1)
    drop_1 = Dropout(0.4)(pool_1)
    
    c2 = Conv3D(filters=16, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(drop_1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(filters=16, kernel_size= (1,3,3), strides=(1,1,1), activation='relu',
                kernel_initializer='glorot_uniform', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    pool_2 = MaxPooling3D(pool_size=(1,2,2))(c2)
    drop_1 = Dropout(0.4)(pool_2)
    
    flat = Reshape((15,6*6*16))(drop_1)
    flat = BatchNormalization()(flat)
    flat = Dropout(0.2)(flat)
    
    lstm_1 = LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(flat)
    
    lstm_2 = LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3)(lstm_1)
    
#    lstm_3 = LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3)(lstm_2)
    
    hidden_1 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(lstm_2)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.4)(hidden_1)
    hidden_2 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Dropout(0.4)(hidden_2)
#    hidden_2a = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
#    hidden_2a = BatchNormalization()(hidden_2a)
#    hidden_2a = Dropout(0.4)(hidden_2a)
    hidden_3 = Dense(256, kernel_initializer='glorot_uniform', activation='relu')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Dropout(0.4)(hidden_3)
#    hidden_4 = Dense(8, kernel_initializer='glorot_uniform', activation='relu')(hidden_3)
#    hidden_4 = BatchNormalization()(hidden_4)
#    hidden_4 = Dropout(0.5)(hidden_4)
    out = Dense(1, activation='linear')(hidden_3)
    
    model = Model(outputs=out, inputs=inp)
    
    print(model.summary())

    return model

from keras.layers import AveragePooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D


def fn_get_model_convLSTM_1():
    
    model = Sequential()
    model.add(AveragePooling3D(pool_size=(1, 4, 4),
                       input_shape=(None,101, 101, 1),
                       padding='same'))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       padding='same', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='linear'))
    
    print(model.summary())

    return model



def fn_get_model_convLSTM_2():
    
    model = Sequential()
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7),
                         input_shape=(None, 101, 101, 1),
                         return_sequences=True,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=16, kernel_size=(7, 7),
                         return_sequences=True,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=8, kernel_size=(7, 7),
                         return_sequences=False,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   data_format='channels_last')) 
    
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='linear'))
    
    print(model.summary())

    return model


def fn_get_model_convLSTM_tframe_1():
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(None, 101, 101, 1),padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='linear',
                   padding='same', data_format='channels_last'))
        
    print(model.summary())
    return model


def fn_get_model_convLSTM_tframe_2():
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       input_shape=(None, 101, 101, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())

#    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
    
    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
        
    print(model.summary())
    
    return model


def fn_get_model_convLSTM_tframe_3():
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                         input_shape=(None, 101, 101, 1), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    
    ### !!! try go_backwards=True !!! ###    
    
    print(model.summary())
    
    return model


def fn_get_model_convLSTM_tframe_4():
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                         input_shape=(None, 101, 101, 1), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())
    
    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    
    ### !!! try go_backwards=True !!! ###    
    
    print(model.summary())
    
    return model

def fn_get_model_convLSTM_tframe_5():
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                         input_shape=(None, 101, 101, 1), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last')) 
    
    print(model.summary())
    
    return model


def fn_run_model(model, X, y, X_val, y_val, batch_size=50, nb_epoch=40,verbose=2,is_graph=False):
    history = History()
    history = model.fit(X, y, batch_size=batch_size, 
                        epochs=nb_epoch,verbose=verbose, validation_data=(X_val, y_val))
    if is_graph:
        fig, ax1 = plt.subplots(1,1)
        ax1.plot(history.history["val_loss"])
        ax1.plot(history.history["loss"])


# -----------------------------------------------------------------------------
# 2D conv

model = fn_get_model()
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss = root_mean_squared_error, optimizer = "adam")

# load data from hd5
X_train, y_train = fn_h5_to_Xy(test_train="train_shuffle",i=0,h_select = my_height, t_select = 14)
X_t_val, y_t_val = fn_h5_to_Xy(test_train="train_shuffle",i=4,h_select = my_height, t_select = 14)

fn_run_model(model, X_train, y_train, X_t_val, y_t_val, batch_size=50,
             nb_epoch=20,verbose=2,is_graph=True)

for i in range(0,4):
    X_train, y_train = fn_h5_to_Xy(test_train="train_shuffle",i=i,h_select = my_height, t_select = 14)
    fn_run_model(model, X_train, y_train, X_t_val, y_t_val, batch_size=50,
             nb_epoch=20,verbose=2,is_graph=False)

# -----------------------------------------------------------------------------
# 2D + timeD

# load data from h5 with 2D + timeD
X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=0,h_select = my_height)
X_t_val, y_t_val = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=4,h_select = my_height)

# compile model
model = fn_get_model_2D_timeD()
model = fn_get_model_2D_LSTM_1()
model = fn_get_model_2D_conv_to_LSTM_1()
model = fn_get_model_convLSTM_1()
model = fn_get_model_convLSTM_2()
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss=fn_keras_rmse, optimizer='adam')

# fit model
fn_run_model(model, X_train, y_train, X_t_val[:100,:,:,:,:], y_t_val[:100,:], batch_size=8, nb_epoch=15,verbose=1,is_graph=True)

# fit on all folds
#for i in range(1,4):
my_height=0
for i in [0,1,2,3,4]:
    fn_print(("training on fold:"+str(i)))
    #X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=i,h_select = my_height)
    #X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle_full",i=i,h_select = my_height)
    X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle_full",i=i,h_select = my_height)
    fn_run_model(model, X_train, y_train, X_t_val[:100,:,:,:,:], y_t_val[:100,:], batch_size=8, nb_epoch=10,verbose=1,is_graph=False)

# -----------------------------------------------------------------------------
# time frame prediction model

from scipy.ndimage import filters
from random import randint

# convert to unaligned timeframes
def fn_Xy_to_tframe(X_in):
    #X0 = X_in[:,0:14,:,:,:]
    #X1 = X_in[:,1:15,:,:,:] # 1 step ahead
    
    # 15 frames makes each sample too large -> samples too slow
    n_frames = 4
    X0 = np.zeros(X_in.shape, dtype=np.float)
    X0 = X0[:,0:n_frames,:,:,:] # only use 5 frames each
    X1 = X0.copy()
    
    # pick 5 random timeframes for each sample and accompanying t+1
    np.random.seed(2)
    for sample in range(0,X_in.shape[0]):
        #rand = np.random.randint(0, 14-n_frames)
        #X0[sample] = X_in[sample,rand:rand+n_frames,:,:,:]
        #X1[sample] = X_in[sample,rand+1:rand+n_frames+1,:,:,:] # 1 step ahead
        
        X1[sample] = X_in[sample,0:5,:,:,:]
        X0[sample] = X_in[sample,1:6,:,:,:]
        
        #X0[sample] = X_in[sample,0::4,:,:,:]
        #X1[sample] = X_in[sample,0::4,:,:,:]
        
        # blur a few inputs randomly
        rand_blur = randint(0, 5)
        for i in range(n_frames):
            if i > rand_blur:
                X0[sample,i,:,:,0] = filters.gaussian_filter(X0[sample,i,:,:,0],sigma=1.8)

                #X0[sample,i,:,:,0] = filters.gaussian_filter(X0[sample,i,:,:,0],sigma=1.8)
                #X1[sample,i,:,:,0] = filters.gaussian_filter(X1[sample,i,:,:,0],sigma=1.4)
    
    # add some noise to X0
    #noise = np.random.normal(0, 0.02, X0.shape)
    #X0 = X0 + noise
    
    # normalise
    X_max = 3.0
    X0 = np.clip(X0,0,X_max) # clip
    X1 = np.clip(X1,0,X_max)
    X0 = X0/X_max
    X1 = X1/X_max
    
    # if want many to one
    #X1 = X1[:,-1] # only last
    #X0 = X0[:,:-1] # excl. last
    
    # quantise to nearest 10th
    #X0 = np.round(X0, 1)
    #X1 = np.round(X1, 1)
    #X0 = np.round(X0/2, 1)*2
    #X1 = np.round(X1/2, 1)*2
    
    return X0,X1

# load data from h5 with 2D + timeD
my_height = 2
X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=0,h_select = my_height)
X_t_val, y_t_val = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=4,h_select = my_height)

X0_train, X1_train = fn_Xy_to_tframe(X_train)
X0_t_val, X1_t_val = fn_Xy_to_tframe(X_t_val)

# for many to one
#X1_train = X1_train[:,4]
#X1_t_val = X1_t_val[:,4]
#X1_train = X1_train.reshape(X1_train.shape[0],1, X1_train.shape[1], X1_train.shape[2], X1_train.shape[3])
#X1_t_val = X1_t_val.reshape(X1_t_val.shape[0],1, X1_t_val.shape[1], X1_t_val.shape[2], X1_t_val.shape[3])

#X0_t_val_all, X1_t_val_all = fn_Xy_to_tframe_all(X_t_val)

# get model
model = fn_get_model_convLSTM_tframe_1()
model = fn_get_model_convLSTM_tframe_2()
model = fn_get_model_convLSTM_tframe_3()
model = fn_get_model_convLSTM_tframe_4()
model = fn_get_model_convLSTM_tframe_5()
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss=fn_keras_rmse, optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam') # doesn't reset weights

# fit model
fn_run_model(model, X0_train, X1_train, X0_t_val[:120,:,:,:,:], X1_t_val[:120,:], batch_size=10, nb_epoch=2,verbose=1,is_graph=True)


# -----------------------------------------------------------------------------
# visualise outputs

# many to many
input_frames = 5
output_frames = input_frames
s_select = 117 # testing on # 3 # 117 # 0 #2 # 1512

X_input = X0_t_val[s_select,:input_frames, :, :, :]
X_true = X1_t_val[s_select,:, :, :, :]

X_pred = model.predict(X_input[np.newaxis, :, :, :, :]) # predict

for i in range(0,output_frames):
    
    # create plot
    fig = plt.figure(figsize=(10, 5))
    
    # truth
    ax = fig.add_subplot(122)
    ax.text(1, -3, ('true tframe:'+str(input_frames+5+i)), fontsize=20, color='b')
    toplot_true = X_true[i, :, :, 0]
    toplot_true[0,0] = 0. # ensure same scale as other
    toplot_true[0,1] = 1.
    plt.imshow(toplot_true,cmap='gist_gray_r')
    
    # predictions   
    ax = fig.add_subplot(121)    
    ax.text(1, -3, ('predictions tframe:'+str(input_frames+5+i)), fontsize=20, color='b')
    toplot_pred = X_pred[0,i, :, :, 0]
    toplot_pred[0,0] = 0. # ensure same scale as other
    toplot_pred[0,1] = 1.
    plt.imshow(toplot_pred,cmap='gist_gray_r')


# for t+1 generate multiple
input_frames = 5
s_select = 25
output_frames = 20
X_input = X0_t_val[s_select,:input_frames, :, :, :]
for i in range(0,output_frames):
    
    X_pred = model.predict(X_input[np.newaxis, :, :, :, :]) # predict
    if i == 0:
        X_save = X_pred[0,4,:,:,0]
        X_save = X_save.reshape(1,101,101,1)
    else:
        X_save = np.concatenate((X_save, X_pred[0,4,:,:,0].reshape(1,101,101,1)), axis=0)
        
    X_input = np.concatenate((X_input, X_pred[0,4,:,:,0].reshape(1,101,101,1)), axis=0)
    X_input = X_input[1:]
X_pred = X_save

# convlstm_1_5_to_2_6_h2_17ep_v01
# is best timeframe+1 model


# for t+1 generate multiple
input_frames = 3
s_select = 25
output_frames = 5
X_input = X0_t_val[s_select,:input_frames, :, :, :]
for i in range(0,output_frames):
    
    X_pred = model.predict(X_input[np.newaxis, :, :, :, :]) # predict
    if i == 0:
        X_save = X_pred[:,:,:,:]
    else:
        X_save = np.concatenate((X_save, X_pred[:,:,:,:]), axis=0)
        
    X_input = np.concatenate((X_input, X_pred[:,:,:,:]), axis=0)
    X_input = X_input[1:]
X_pred = X_save

    
for i in range(0,output_frames):
    # create plot
    fig = plt.figure(figsize=(10, 5))
    
    # predictions   
    ax = fig.add_subplot(121)    
    ax.text(1, -3, ('predictions tframe:'+str(input_frames+5+i)), fontsize=20, color='b')
    toplot_pred = X_pred[i, :, :, 0]
    toplot_pred[0,0] = 0. # ensure same scale as other
    toplot_pred[0,1] = 1.
    plt.imshow(toplot_pred,cmap='gist_gray_r')
    
    

    
    

# scatter graph 
results_pred = []
results_pred2 = []
results_pred3 = []
results_true = []
target = np.arange(30,71)
target2 = np.arange(40,61)
target3 = np.arange(50,51)

for s_select in range(1000):
    if s_select % 400==0:
        fn_print(("predicting: "+str(s_select)))
        
    # predictions
    #X_input = X0_t_val[s_select,:input_frames, :, :, :] # to assess forecast quality
    X_input = X1_t_val[s_select,:input_frames, :, :, :] # carry fw for rain predictions
    X_pred = model.predict(X_input[np.newaxis, :, :, :, :]) # predict
    toplot_pred = X_pred[0,4, :, :, 0]
    #toplot_pred = X0_t_val[s_select,4, :, :, :] # for last input
    sum_mid_pred = np.average(toplot_pred[target,target])
    sum_mid_pred2 = np.average(toplot_pred[target2,target2])
    sum_mid_pred3 = np.average(toplot_pred[target3,target3])
    results_pred.append(sum_mid_pred)
    results_pred2.append(sum_mid_pred2)
    results_pred3.append(sum_mid_pred3)
    
    # truth
    true_toplot = X1_t_val[s_select, 4, :, :, 0]
    sum_mid_true = np.average(true_toplot[target,target])
    results_true.append(sum_mid_true)

fig, ax1 = plt.subplots(1,1)
ax1.scatter(results_true,results_pred,alpha=0.5,s=2,color="blue")
ax1.scatter(results_true,results_pred2,alpha=0.5,s=2,color="r")
ax1.scatter(results_true,results_pred3,alpha=0.5,s=2,color="k")
ax1.set_xlabel('X_true')
ax1.set_ylabel('X_pred')

corr_score = np.corrcoef((results_true, results_pred), rowvar=True)
corr_score2 = np.corrcoef((results_true, results_pred2), rowvar=True)
corr_score3 = np.corrcoef((results_true, results_pred3), rowvar=True)
print("corr_score:", np.round(corr_score,3))
print("corr_score2:", np.round(corr_score2,3))
print("corr_score3:", np.round(corr_score3,3))


# so for predictions 10 timeframes (1hr) ahead
# correlation first 800
# if use last frame
# 0.631 - (40,61)
# 0.550 - (50,51)
# using train_shuffle_full 0
# 0.700 - (30,71)
# 0.647 - (40,61)
# 0.544 - (50,51)

# ~8ep
# convlstm_1_5_to_10_15_8ep_v01
# 0.871 - (40,61)

# ~10ep
# convlstm_1_5_to_10_15_10ep_v02
# 0.871 - (40,61)
# full 1000
# 0.855 - (40,61)
# 0.842 - (50,51)

# convlstm_1_5_to_10_15_10ep_v02
# 0.871 - (40,61)
# full 1000
# 0.855 - (40,61)
# 0.842 - (50,51)
# using train_shuffle 0
# 0.900 - (30,71)
# 0.890 - (40,61)
# using train_shuffle_full 0
# 0.897 - (30,71)
# 0.886 - (40,61)
# 0.858 - (50,51)
# using train 7
# 0.935 - (30,71)
# 0.921 - (40,61)
# 0.883 - (50,51)

# height = 1
# convlstm_1_5_to_10_15_h1_12ep_v02
# train_shuffle 4
# 0.901 - (30,71)
# 0.891 - (40,61)
# 0.862 - (50,51)

# find rela between predicted map and actual rainfall

rainfall = y_t_val[:1000].reshape(-1)

fig, ax1 = plt.subplots(1,1)
ax1.scatter(results_pred, rainfall,alpha=0.5,s=2,color="b")
ax1.scatter(results_pred2, rainfall,alpha=0.5,s=2,color="r")
ax1.scatter(results_pred3, rainfall,alpha=0.5,s=2,color="k")
ax1.set_xlabel('X_pred')
ax1.set_ylabel('Rainfall_true')

corr_score = np.corrcoef((rainfall, results_pred), rowvar=True)
corr_score2 = np.corrcoef((rainfall, results_pred2), rowvar=True)
corr_score3 = np.corrcoef((rainfall, results_pred3), rowvar=True)
print("corr_score:", np.round(corr_score,3))
print("corr_score2:", np.round(corr_score2,3))
print("corr_score3:", np.round(corr_score3,3))

# using train 1
# 0.110 - (40,61)
# 0.104 - (50,51)

# using train_shuffle 0
# -0.031 - (30,71)
# -0.029 - (40,61)

# using train_shuffle_full 0
# -0.093 - (30,71)
# -0.077 - (40,61)
# -0.074 - (50,51)

# using train 7
# -0.147 - (30,71)
# -0.165 - (40,61)
# -0.172 - (50,51)

# height = 1
# convlstm_1_5_to_10_15_h1_12ep_v02
# train_shuffle 4
# -0.088 - (30,71)
# -0.082 - (40,61)
# -0.088 - (50,51)

plt.figure(10)
plt.hist(y_t_val,bins=100)



# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions]
def fn_blur(X_in):
    X = filters.gaussian_filter(X_in,sigma=1)
    return X

# 10 ahead

input_frames = 5
output_frames = 1
s_select = 1500 # testing on # 3 # 117
#track = X0_t_val[s_select,:input_frames, :, :, :]


X_input_t_val = X0_t_val[s_select,:input_frames, :, :, :]
X_pred_t_val = model.predict(X_input_t_val[np.newaxis, :, :, :, :]) # predict
#X_pred_t_val = fn_blur(X_pred_t_val)
X_pred_t_val = (X_pred_t_val-0.05)#*8/6 # manual transformation
X_pred_t_val = np.round(X_pred_t_val,1)
#X_pred_t_val = np.round(X_pred_t_val/2, 1)*2
X_pred_t_val = np.clip(X_pred_t_val,0,1)

# And then compare the predictions
# to the ground truth
for i in range(output_frames):
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    
    # initial input / predictions       
    ax.text(1, -3, ('predictions tframe:'+str(input_frames+5+i)), fontsize=20, color='b')

    #toplot = X_pred_t_val[0,i, :, :, 0] # output 5
    toplot = X_pred_t_val[0, :, :, 0] # output last 1
    #toplot = np.round(toplot, 1)
    plt.imshow(toplot,cmap='gist_gray_r')
    sum_mid = np.sum(toplot[40:61,40:61])
    print("predicted density:",round(sum_mid,2))
    
    # truth
    ax = fig.add_subplot(122)
    
    ax.text(1, -3, ('true tframe:'+str(input_frames+5+i)), fontsize=20, color='b')

    #toplot = X1_t_val[s_select,i, :, :, 0]
    toplot = X1_t_val[s_select, :, :, 0]
    #toplot = np.round(toplot, 1)
    plt.imshow(toplot,cmap='gist_gray_r')
    sum_mid = np.sum(toplot[40:61,40:61])
    print("actual density:",round(sum_mid,2))

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.hist(X_pred_t_val.reshape(-1),bins=100)
ax.set_xlim([-0.02,1.02])
ax = fig.add_subplot(122)
#ax.hist(X1_t_val[s_select,:, :, :, 0].reshape(-1),bins=100)
ax.hist(X1_t_val[s_select, :, :, 0].reshape(-1),bins=100)
ax.set_xlim([-0.02,1.02])


# scatter graph of 10 time frame predictions!
results_pred = []
results_true = []
for s_select in range(1600):
    if s_select % 400==0:
        fn_print(("predicting: "+str(s_select)))
    #X_input_t_val = X0_t_val[s_select,:input_frames, :, :, :]
    X_input_t_val = X0_train[s_select,:input_frames, :, :, :]
    X_pred_t_val = model.predict(X_input_t_val[np.newaxis, :, :, :, :]) # predict
    pred_toplot = X_pred_t_val[0, :, :, 0]
    #pred_toplot = X0_t_val[s_select,4, :, :, :]
    sum_mid_pred = np.sum(pred_toplot[40:61,40:61])
    
    true_toplot = X1_t_val[s_select, :, :, 0]
    sum_mid_true = np.sum(true_toplot[40:61,40:61])
    results_pred.append(sum_mid_pred)
    results_true.append(sum_mid_true)

fig, ax1 = plt.subplots(1,1)
ax1.scatter(results_true,results_pred,alpha=0.5,s=2)
ax1.set_xlabel('X_true')
ax1.set_ylabel('X_pred')

corr_score = np.corrcoef((results_true, results_pred), rowvar=True)
print("corr_score:", corr_score)

# model_convlstm_tframe_1_5to10_q10_mto1_v08
# trained on binary cross entopy
# 0.696 - 40 to 61
# 0.571 - 50 to 51

# trained 5 eps on mse
# 0.694 - 40 to 61
# 0.569 - 50 to 51

# compared with last timeframe (10 before)
# 0.676 - 40 to 61
# 0.525 - 50 to 51

# not quantised
# TO DO

# do linear regression model on training data
# find rela between results_pred and y_train

# visualise last output
fig, ax1 = plt.subplots(1,1)
plt.imshow(pred_toplot,cmap='gist_gray_r')

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.hist(pred_toplot.reshape(-1),bins=100)


freeze_last = X_train[:,14,50,50,0]

avg_last = X_train[:,14,40:61,40:61,0]
result_avg_X = []
for i in range(avg_last.shape[0]):
    avg_X = avg_last[i].mean()
    result_avg_X.append(avg_X)
    


fig, ax1 = plt.subplots(1,1)
ax1.scatter(result_avg_X,y_train,alpha=0.5,s=2)
#ax1.scatter(results_pred,y_train,alpha=0.5,s=2)
ax1.set_xlabel('X_value')
ax1.set_ylabel('y_rainfall_true')

corr_score = np.corrcoef((result_avg_X,y_train.reshape(-1)), rowvar=True)
#corr_score = np.corrcoef((results_pred,y_train.reshape(-1)), rowvar=True)
print("corr_score:", corr_score)

# -0.033 - 50 to 51
# -0.035 - 40 to 61

# train i=0
# 0.042 - 0:101
# 0.024 - 40 to 61

# linear model based off last frame
from sklearn import linear_model

# load data
X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle_full",i=0,h_select = my_height)
X_t_val, y_t_val = fn_h5_to_Xy_2D_timeD(test_train="testB",i=0,h_select = my_height)
X_t_val, y_t_val = fn_h5_to_Xy_2D_timeD(test_train="train",i=9,h_select = my_height)

lr_model = linear_model.LinearRegression()

lr_X_train = X_train[:,14,:,:,0]
lr_X_train = lr_X_train.reshape(lr_X_train.shape[0],-1)

lr_X_val = X_t_val[:,14,:,:,0]
lr_X_val = lr_X_val.reshape(lr_X_val.shape[0],-1)

lr_y_train = y_train
lr_y_val = y_t_val

# fit model
lr_model.fit(lr_X_train, lr_y_train)

# test on val data
lr_pred = lr_model.predict(lr_X_val)
lr_pred = np.clip(lr_pred,0,lr_pred.max()) # can't be zero
lr_rmse = np.square(np.mean((lr_pred - lr_y_val) ** 2))
print("lr_rmse",lr_rmse)

# plot distributions
fig, ax1 = plt.subplots(1,1)
ax1.hist(lr_pred,bins=100)
ax1.set_ylabel('y_rainfall_pred')

fig, ax1 = plt.subplots(1,1)
ax1.hist(lr_y_val,bins=100)
ax1.set_ylabel('y_rainfall_true')

# fitted on train_shuffle 0
# rmse_train_8 = 5.76
# rmse_train_9 = 4.22


# do real predictions
# linear regression

# find model linking intensity sum to rainfall

# try on tranch 8!


# time step +1

input_frames = 5
output_frames = 4
s_select = 3 # testing on # 3 # 117
track = X0_t_val[s_select,:input_frames, :, :, :]

for j in range(output_frames):
    #model.reset_states() # makes no difference
    new_pos = model.predict(track[np.newaxis, :, :, :, :]) # predict
    
    new = new_pos[:, -1, :, :, :] # select the last frame output
    #new = fn_blur(new)
    #new = filters.median_filter(new, 3)
    #new = new-0.1
    #new = np.round(new, 1)
    new = np.clip(new,0,1)
    track = np.concatenate((track, new), axis=0) # append to track with new frame (and loop)
    track = track[1:] # drop the first frame

for i in range(track.shape[0]):
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    
    # initial input / predictions
    if i >= 5 - output_frames:
        ax.text(1, -3, ('Predictions tfrm:'+str(i)), fontsize=20, color='r')
    else:
        ax.text(1, -3, 'Inital trajectory', fontsize=20, color='b')

    toplot = track[i, :, :, 0]
    #toplot = np.round(toplot, 1)
    plt.imshow(toplot,cmap='gist_gray_r')


# And then compare the predictions
# to the ground truth
for i in range(input_frames+output_frames):
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    
    # initial input / predictions
    if i >= input_frames:
        ax.text(1, -3, ('Predictions tfrm:'+str(i)), fontsize=20, color='r')
    else:
        ax.text(1, -3, 'Inital trajectory', fontsize=20, color='b')

    toplot = track[i, :, :, 0]
    #toplot = np.round(toplot, 1)
    plt.imshow(toplot,cmap='gist_gray_r')
    
    # truth
    ax = fig.add_subplot(122)
    if i > X1_t_val.shape[1]:
        plt.text(1, -3, 'No ground truth', fontsize=20, color='r')
        plt.imshow(np.zeros((101,101)),cmap='gist_gray_r')
    else:
        plt.text(1, -3, 'Ground truth', fontsize=20, color='b')
        if i < 2:
            toplot = X0_t_val[s_select,i, :, :, 0]
        else:
            toplot = X1_t_val[s_select,i - 1, :, :, 0]
        #toplot = np.round(toplot, 1)
        plt.imshow(toplot,cmap='gist_gray_r')
    
    #plt.savefig('%i_animate.png' % (i + 1))
    

s_frame = 2
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.text(1, 3, 'predicted', fontsize=10, color='b')
ax.hist(track[s_frame,:,:,0].reshape(-1),bins=100)
ax = fig.add_subplot(122)
ax.text(1, 3, 'actual', fontsize=10, color='b')
ax.hist(X1_t_val[s_select,s_frame-1, :, :, 0].reshape(-1),bins=100)

# visualise new_pos
for i in range(3):
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    
    # initial input / predictions
    toplot = new_pos[0,i, :, :, 0]
    #toplot = np.round(toplot, 1)
    plt.imshow(toplot,cmap='gist_gray_r')


# time frame + 1, many to one
input_frames = 5
output_frames = 5
s_select = 117 # testing on # 3 # 117
track = X0_t_val[s_select,:input_frames, :, :, :]

for j in range(output_frames):
    new_pos = model.predict(track[np.newaxis, :, :, :, :]) # predict
    new = new_pos[:, :, :, :] # select the last frame output
    #new = fn_blur(new)
    #new = filters.median_filter(new, 3)
    #new = new-0.1
    #new = np.round(new, 1)
    #new = np.clip(new,0,1)
    track = np.concatenate((track, new), axis=0) # append to track with new frame (and loop)


# And then compare the predictions
# to the ground truth
for i in range(input_frames+output_frames):
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    
    # initial input / predictions
    if i >= input_frames:
        ax.text(1, -3, ('Predictions tfrm:'+str(i)), fontsize=20, color='r')
    else:
        ax.text(1, -3, 'Inital trajectory', fontsize=20, color='b')

    toplot = track[i, :, :, 0]
    #toplot = np.round(toplot, 1)
    plt.imshow(toplot,cmap='gist_gray_r')
    
    # truth
    ax = fig.add_subplot(122)
    if i > X1_t_val.shape[1]:
        plt.text(1, -3, 'No ground truth', fontsize=20, color='r')
        plt.imshow(np.zeros((101,101)),cmap='gist_gray_r')
    else:
        plt.text(1, -3, 'Ground truth', fontsize=20, color='b')
        if i < input_frames:
            toplot = X0_t_val[s_select,i, :, :, 0]
            plt.imshow(toplot,cmap='gist_gray_r')
        elif i == input_frames:
            toplot = X1_t_val[s_select, :, :, 0]
            plt.imshow(toplot,cmap='gist_gray_r')
        #toplot = np.round(toplot, 1)
        
    

# -----------------------------------------------------------------------------
# evaluate model

def fn_examine_output(y_true, y_pred):
    
    # un normalise outputs
    y_true_unnorm = y_true*y_std # + y_mean
    y_pred_unnorm = y_pred*y_std # + y_mean
    #y_pred_unnorm = y_pred_unnorm.clip(0,y_pred_unnorm.max())
    
    # scatter plot
    fig, ax1 = plt.subplots(1,1)
    ax1.scatter(y_true_unnorm.reshape(-1),y_pred_unnorm.reshape(-1),alpha=0.5,s=2)
    ax1.set_ylabel('y_pred')
    ax1.set_xlabel('y_true')
    
    # distribution
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.hist(y_true_unnorm,bins=100)
    ax1.set_title("y_true")
    ax2.hist(y_pred_unnorm,bins=100)
    ax2.set_title("y_pred")
    
    # correlation
    #corr_score = np.corrcoef((y_true_unnorm, y_pred_unnorm), rowvar=True)
    #print("corr_score:", corr_score)
    
    # rmse score
    #y_pred_unnorm = np.clip(y_pred_unnorm,0,y_pred_unnorm.max())
    rmse_score = fn_rmse(y_true_unnorm, y_pred_unnorm)
    print("rmse_score:", np.round(rmse_score,2))
    
    # mse score
    #mse_score = fn_mse(y_true_unnorm, y_pred_unnorm)
    #print("mse_score:", np.round(mse_score,2))
    return rmse_score
    
    

# load data to test
X_output = X_t_val
y_true = y_t_val

# OR
#X_test, y_test = fn_h5_to_Xy(test_train="train",i=8,h_select = my_height, t_select = 14)
X_test, y_test = fn_h5_to_Xy_2D_timeD(test_train="train",i=8,h_select = my_height)

for i in [8,9]:
    X_test, y_test = fn_h5_to_Xy_2D_timeD(test_train="train",i=i,h_select = my_height)
    X_output = X_test
    y_true = y_test
    
    # make predictions
    y_pred = model.predict(X_output, batch_size=30,verbose=1)
    #model.evaluate(X_output, y_true, batch_size=30)
    #fn_rmse(y_true*y_std, y_pred*y_std)
    
    # print output
    rmse_score = fn_examine_output(y_true, y_pred)
    print("\n rmse grp ",i," =", np.round(rmse_score,2))
    



# -----------------------------------------------------------------------------
# save or load model

# save
def fn_save_model(model, model_name):
    model_json = model.to_json()
    model_name = model_name
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")

# load model
from keras.models import model_from_json
def fn_load_model(model_name):
    json_file = open(model_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    return loaded_model

fn_save_model(model, "model_lstm1_trainfull_h0_v01")

model = fn_load_model("model_lstm1_trainfull_h1_v01")
print(model.summary())

# -----------------------------------------------------------------------------
# create submission

# load data
X_testA, y_testA = fn_h5_to_Xy(test_train="testA",i=0,h_select = my_height, t_select = 14)
X_testA, y_testA = fn_h5_to_Xy_2D_timeD(test_train="testB",i=0,h_select = my_height)

# do prediction
y_pred = model.predict(X_testA, batch_size=8)
#y_pred = lr_pred # linear model

# un normalise output
y_pred_unnorm = y_pred*y_std # + y_mean
y_pred_unnorm = y_pred_unnorm.clip(0,y_pred_unnorm.max())

# check looks sensible
fig, (ax1 ,ax2)= plt.subplots(2,1)
ax1.hist(y_pred_unnorm,bins=100)
ax1.set_title("y_pred histogram")
ax2.plot(y_pred_unnorm)
ax2.set_title("y_pred time series")
if y_pred.shape[0] != 2000:
    print("ERROR CHECK PREDICTION LENGTH")

# write file
filename_out = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\output\\Phase_B_pred_" + datetime.now().strftime('%m_%d_%H_%M') + ".csv"
np.savetxt(filename_out, y_pred_unnorm)



"""
linear model
# fitted on train_shuffle 0
# rmse_train_8 = 5.76
# rmse_train_9 = 4.22

# fitted on train_shuffle_full 0
# ldrboard = 19.9 (testB)

"""

# merge files
from numpy import genfromtxt
filename_in_h0 = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\output\\Phase_B_pred_06_30_23_29 h0.csv"
filename_in_h1 = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\output\\Phase_B_pred_06_30_22_53 h1.csv"
filename_in_h2 = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\output\\Phase_B_pred_06_30_22_31 h2.csv"
filename_in_h3 = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\output\\Phase_B_pred_06_30_22_14 h3.csv"
h0 = genfromtxt(filename_in_h0, delimiter=',')
h1 = genfromtxt(filename_in_h1, delimiter=',')
h2 = genfromtxt(filename_in_h2, delimiter=',')
h3 = genfromtxt(filename_in_h3, delimiter=',')
 
h_all = np.zeros(2000)
h_all = h_all.reshape(-1,1)
for i in range(h0.shape[0]):
    h_all[i] = (h0[i] + h1[i] + h2[i] + h3[i])/4
    
y_pred_unnorm = h_all

# -----------------------------------------------------------------------------
# junk

# X input values vs rain - can we binarise?
from scipy.stats.stats import pearsonr 
X_train0, y_train0 = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=0,h_select = 3)
X_train0, y_train0 = fn_h5_to_Xy_2D_timeD(test_train="train",i=8,h_select = 1)

X_train0 = X_train0[:, 14, :, :, 0]

avg_X = []
avg_X_nz = []
max_X = []
rain = []
for i in range(X_train0.shape[0]):
    if (X_train0[i, :, :] > 0).sum() > 2000: # more than 10% are clouds
        avg_X.append(X_train0[i, :, :].mean())
        avg_X_nz.append(X_train0[i, :, :][(X_train0[i, :, :] > 0)].mean())
        max_X.append(X_train0[i, :, :].max())
        rain.append(y_train0[i][0])
    
fig, ax1 = plt.subplots(1,1)
ax1.scatter(avg_X,rain,alpha=0.5,s=2)
ax1.set_xlabel("avg_X")
ax1.set_ylabel("rain")
ax1.set_xlim([0., 2.1])
print("avg_X",pearsonr(avg_X,rain))

fig, ax1 = plt.subplots(1,1)
ax1.scatter(avg_X_nz,rain,alpha=0.5,s=2)
ax1.set_xlabel("avg_X_nz")
ax1.set_ylabel("rain")
ax1.set_xlim([0., 2.1])
print("avg_X_nz",pearsonr(avg_X_nz,rain))

fig, ax1 = plt.subplots(1,1)
ax1.scatter(max_X,rain,alpha=0.5,s=2)
ax1.set_xlabel("max_X")
ax1.set_ylabel("rain")
print("max_X",pearsonr(max_X,rain))
  
   



# analysis of 0 train shuffled vs 8 train
X_train0, y_train0 = fn_h5_to_Xy_2D_timeD(test_train="train_shuffle",i=0,h_select = 1)
X_train8, y_train8 = fn_h5_to_Xy_2D_timeD(test_train="train",i=8,h_select = 1)

no_0 = y_train0.reshape(-1).shape

#no_0_X = int(np.floor(X_train0.reshape(-1).shape[0]/1.6))
#no_8_X = X_train8.reshape(-1).shape[0]

fig, (ax1 ,ax2)= plt.subplots(2,1)
ax1.hist(X_train0.reshape(-1)[0:int(np.floor(X_train0.reshape(-1).shape[0]/1.6))],bins=100)
ax1.set_title("X_train0 histogram")
ax1.set_xlim([-1.5, 3]) 
ax2.hist(X_train8.reshape(-1),bins=100)
ax2.set_title("X_train8 histogram")
ax2.set_xlim([-1.5, 3]) 

fig, (ax1 ,ax2)= plt.subplots(2,1)
ax1.hist(y_train0.reshape(-1)[0:int(np.floor(y_train0.reshape(-1).shape[0]/1.6))],bins=100)
ax1.set_title("y_train0 histogram")
ax1.set_xlim([0, 6]) 
ax1.set_ylim([0, 160]) 
ax2.hist(y_train8.reshape(-1),bins=100)
ax2.set_title("y_train8 histogram")
ax2.set_xlim([0, 6]) 
ax2.set_ylim([0, 160]) 

y_train0.mean()
y_train8.mean()


plt.hist(X_testA.reshape(-1),bins=100)

for i in [0,3,4,8,9]:#range(5,10):
    X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train",i=i,h_select = my_height)
    plt.figure(i+10)
    plt.hist(y_train.reshape(-1),bins=100)
### X ###
### FOLD 4 AND 8 IS MOST SIMILAR TO TRAINING DATA ### (starting at 0)
### many more zeros for testA ###



# can we profit from seasonality in dataset? no
for i in range(0,10):
    np_train_id, np_train_label, np_train_data  = fn_h5_to_np(test_train="train",i=i)
    X, y = fn_np_to_Xy(np_train_data, np_train_label, h_select = 2, t_select = 14)
    X, y = fn_np_to_Xy(np_test_data, np_test_label, h_select = 2, t_select = 14)
    y_std = y.reshape(-1).std()
    y_mean = y.reshape(-1).mean()
    X_std = X.reshape(-1).std()
    X_mean = X.reshape(-1).mean()
    print("group", i, "\tstd", X_std, "\tmean", X_mean)

# y
#0       std 13.2610919973       mean 12.2648
#1       std 16.8392921039       mean 22.0988
#2       std 17.3752807721       mean 16.5197,
#3       std 14.731613359        mean 12.7028,
#4       std 15.9536324278       mean 16.3606,
#5       std 16.7297267279       mean 17.0231,
#6       std 14.1466991203       mean 15.748,
#7       std 14.947103117        mean 13.8721,
#8       std 17.2471783791       mean 17.4714,
#9       std 13.8084248454       mean 11.3927,

#testA
#y_std
#15.933802081110459
#y_mean
#17.181799999999999

result = [12.2648,22.0988,16.5197, 12.7028,16.3606,17.0231,15.748,13.8721,17.4714,11.3927]
plt.plot(result)

# X
#group 0         std 43.2829508117       mean 86.3431349868
#group 1         std 51.8117621949       mean 67.2934060386
#group 2         std 52.3563855296       mean 63.5740025488
#group 3         std 49.8259649493       mean 58.1280911675
#group 4         std 50.5309812121       mean 63.1805846486
#group 5         std 49.1286819389       mean 63.8588425645
#group 6         std 45.5899244315       mean 52.0425168121
#group 7         std 48.9337589231       mean 48.4350874424
#group 8         std 44.4677307727       mean 44.0360308793
#group 9         std 40.2562715732       mean 63.6335146554
#testA           std 48.6791465579       mean 76.8182705127

path_in_train = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\data_new\\CIKM2017_train\\train.txt"
#path_in_train = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\data_new\\CIKM2017_train\\data_sample.txt"
np_train_id, np_train_label, np_train_data = fn_load_data(path_in_train, start_line=5, no_import=5)
# starts from after start_line

path_in_test = "D:\\02 Datasets\\05 Precipitation Nowcasting data\\data_new\\CIKM2017_testA\\testA.txt"
# np_test_id, np_test_label, np_test_data = fn_load_data(path_in, path_in_test=True)
# dimensions: np_train_data[sample, time, height, Y, X]

# -----------------------------------------------------------------------------
# testing new rmse loss fn

from keras import backend as K
from tensorflow.contrib.keras import backend as K2

model.compile(loss=fn_keras_rmse, optimizer='adam')
fn_run_model(model, X_train, y_train, X_t_val, y_t_val, batch_size=30, nb_epoch=3,verbose=1,is_graph=False)

model.fit(X_train, y_train, batch_size=30, epochs=1,verbose=1, validation_data=(X_t_val, y_t_val))


fn_keras_rmse(y_true, y_pred).eval()
fn_keras_rmse2(y_true*y_std, y_pred*y_std).eval()
fn_rmse(y_true*y_std, y_pred*y_std)
fn_rmse(y_true, y_pred)

def fn_keras_rmse2(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred) - (y_true))))
    #return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def fn_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def fn_mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

import tensorflow as tf
sess = tf.InteractiveSession()
temp = K.square(y_pred - y_true) # 1600, 1
temp = K.mean(K.square((y_pred) - (y_true))) # 3.9796
temp = K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std)))) # 2.821 -> but np.sqrt(3.979) = 1.994

temp = y_pred - y_true
temp = K.square(temp)

temp = K.square(4)

temp.eval()
out = temp.eval()
print(out)

np.mean(out)
np.sqrt(np.mean(out))

np.mean(K.square(y_pred - y_true)) # 3.979
np.sqrt(np.mean(np.square((y_pred*y_std) - (y_true*y_std)))) # 1.989

sess.close()

# so it's K.square that's the problem
# and K.sqrt I think

# np.sqrt(3.979) = 1.994
# K.sqrt(3.979) = 2.821

diff = y_pred - y_true 
array([[-0.6       ],
       [-1.4       ],
       [-0.27333333],
       ..., 
       [-2.13530654],
       [-0.15333333],
       [-0.6       ]])

out = K.square(diff)
array([[ 0.72      ],
       [ 3.92      ],
       [ 0.14942222],
       ..., 
       [ 9.11906807],
       [ 0.04702222],
       [ 0.72      ]])

np.square(diff)
array([[ 0.36      ],
       [ 1.96      ],
       [ 0.07471111],
       ..., 
       [ 4.55953403],
       [ 0.02351111],
       [ 0.36      ]])
    
def fn_keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std))))
    

fn_keras_rmse2(y_true*y_std, y_pred*y_std).eval()
fn_keras_rmse(y_true, y_pred).eval()
fn_rmse(y_true*y_std, y_pred*y_std)

# -----------------------------------------------------------------------------
# comparing speed

data_temp=np.arange(0,50000000)
data_temp_str = [str(x) for x in data_temp]

fn_print("method 1 start")
data_temp = [int(x) for x in data_temp_str]
fn_print("method 1 end")

fn_print("method 2 start")
data_temp = list(map(int, data_temp_str))
fn_print("method 2 end")


# -----------------------------------------------------------------------------
# practise
# http://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# ---------------------------------------------------------------
# A - one to one
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(len(seq), 1, 1)
y = seq.reshape(len(seq), 1)
# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(1, input_shape=(1, 1)))
#model.add(Dense(3))
#model.add(Dense(3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, nb_epoch=n_epoch, batch_size=n_batch, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)
    
# ---------------------------------------------------------------
# B - many to one
# prepare sequence
length = 5
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 100
# create LSTM
model = Sequential()
model.add(LSTM(7, input_shape=(length, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, nb_epoch=n_epoch, batch_size=n_batch, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
print('result', result)
print('shape', result.shape)


# tp example - 5 time steps, 2 features
length = 5
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 100
# create LSTM
model = Sequential()
model.add(LSTM(7, input_shape=(length, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, nb_epoch=n_epoch, batch_size=n_batch, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
print('result', result)
print('shape', result.shape)

