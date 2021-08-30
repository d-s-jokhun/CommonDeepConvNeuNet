#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


# In[ ]:


def Ch_Norm_255(X, Saturation = [0,0]):
#     Offsets and rescales each channel such that the intensities in each channel varies between 0 and 255
# Incoming 0 values are output as 0s
    # assert len(X.shape)==4, 'Channel_GrayScale expects a 4 dimensional tensor (batch_size,img_height,img_width,nCh)'
    bw = tf.dtypes.cast(X!=0, tf.float32)
    bw_inv = tf.dtypes.cast(X==0, tf.float32)
    bckgrnd_max = bw_inv * tf.keras.backend.max(X, axis=[1,2], keepdims=True)
    
    def minim (img, Saturation):
        return (np.percentile(img,100*(Saturation), axis=[1,2], keepdims=True)).astype(np.float32)

    offset = tf.numpy_function(minim, [X+bckgrnd_max, Saturation[0]], tf.float32)

    X = (tf.dtypes.cast(X, tf.float32) - offset)*bw

    def saturate_scale (X, Saturation):
        maxim = (np.percentile(X,100*(1-Saturation), axis=[1,2], keepdims=True)).astype(np.float32)
        return np.clip((X/maxim)*255,0,255)

    scaled = tf.numpy_function(saturate_scale, [X, Saturation[1]], tf.float32)

    return tf.dtypes.cast(scaled, tf.float32)

