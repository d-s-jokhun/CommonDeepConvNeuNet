#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


def Ch_Norm_255(X):
#     Offsets and rescales each channel such that the intensities in each channel varies between 0 and 255
# Incoming 0 values are output as 0s
    assert len(X.shape)==4, 'Channel_GrayScale expects a 4 dimensional tensor (batch_size,img_height,img_width,nCh)'
    bw = tf.dtypes.cast(X!=0, X.dtype)
    bw_inv = tf.dtypes.cast(X==0, X.dtype)

    maxi = tf.keras.backend.max(X, axis=[1,2], keepdims=True)
    bckgrnd_max = bw_inv * maxi
    offset = tf.keras.backend.min(X+bckgrnd_max, axis=[1,2], keepdims=True)
    X = (X-offset)*bw

    scale = 255/tf.keras.backend.max(X, axis=[1,2], keepdims=True)
    X = tf.dtypes.cast(X*scale, tf.float32)
    
    return X

