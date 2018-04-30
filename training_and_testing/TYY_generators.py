import keras
import numpy as np
import sys
from scipy import misc
import tensorflow as tf


def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    w = x.shape[0]
    h = x.shape[1]
    out = x[0+dx:w-(dn-dx),0+dy:h-(dn-dy),:]
    out = misc.imresize(out, (w,h), interp='nearest')
    return out

def augment_data(images):
    for i in range(0,images.shape[0]):

        if np.random.random() > 0.5:
            images[i] = images[i][:,::-1]
        """
        if np.random.random() > 0.5:
            images[i] = random_crop(images[i],4)
        """
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_rotation(images[i], 20, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_shear(images[i], 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_shift(images[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_zoom(images[i], [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        
    return images


def data_generator_reg(X,Y,batch_size):

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p,q = [],[]
        for i in range(len(X)):
            p.append(X[i])
            q.append(Y[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),np.array(q)
                p,q = [],[]
        if p:
            yield augment_data(np.array(p)),np.array(q)
            p,q = [],[]

def data_generator_dex(X,Y,batch_size):

    Y1 = Y[0]
    Y2 = Y[1]

    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        p,q1,q2 = [],[],[]
        for i in range(len(X)):
            p.append(X[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            if len(p) == batch_size:
                yield augment_data(np.array(p)),[np.array(q1),np.array(q2)]
                p,q1,q2 = [],[],[]
        if p:
            yield augment_data(np.array(p)),[np.array(q1),np.array(q2)]
            p,q1,q2 = [],[],[]

def data_generator_dex_centerloss(X,Y,batch_size):
    X1 = X[0]
    X2 = X[1]
    Y1 = Y[0]
    Y2 = Y[1]
    Y3 = Y[2]
    while True:
        idxs = np.random.permutation(len(X1))
        X1 = X1[idxs] #images
        X2 = X2[idxs] #labels for center loss
        Y1 = Y1[idxs]
        Y2 = Y2[idxs]
        Y3 = Y3[idxs]
        p1,p2,q1,q2,q3 = [],[],[],[],[]
        for i in range(len(X1)):
            p1.append(X1[i])
            p2.append(X2[i])
            q1.append(Y1[i])
            q2.append(Y2[i])
            q3.append(Y3[i])
            if len(p1) == batch_size:
                yield [augment_data(np.array(p1)),np.array(p2)],[np.array(q1),np.array(q2),np.array(q3)]
                p1,p2,q1,q2,q3 = [],[],[],[],[]
        if p1:
            yield [augment_data(np.array(p1)),np.array(p2)],[np.array(q1),np.array(q2),np.array(q3)]
            p1,p2,q1,q2,q3 = [],[],[],[],[]