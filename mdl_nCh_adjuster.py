#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import Lambda_functions


# In[3]:


def mdl_nCh_adjuster (NumOfInputCh=1, NumOfOutputCh=3, InputImgSize=None, OutputImgSize=None):
    
    assert (type(InputImgSize)!=type(None) and InputImgSize!=[] and InputImgSize!=()), 'InputImgSize has to be a pair or values denoting the height and width of the input!'
    
    if OutputImgSize==[] or OutputImgSize==() or type(OutputImgSize)==type(None):
        OutputImgSize=InputImgSize
        
        
    In = tf.keras.Input(shape=(InputImgSize[0], InputImgSize[1], NumOfInputCh), name="Input")

    x = tf.keras.layers.Lambda(
        lambda image: tf.image.resize_with_pad(
            image, target_height=OutputImgSize[0], target_width=OutputImgSize[1], 
            method=tf.image.ResizeMethod.BILINEAR, antialias=True), name="Resizing")(In)
    
    x = tf.keras.layers.Conv2D(
        filters=NumOfOutputCh, kernel_size=1, strides=(1, 1), padding="same",data_format=None, 
        activation=None, use_bias=False, kernel_initializer="glorot_uniform", bias_initializer="zeros",
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, bias_constraint=None, name="SinglePixConv")(x) 
    
    Out=tf.keras.layers.Lambda(
        Lambda_functions.Channel_GrayScale, name='Ch_GrayScale')(x)
    
    mdl_nCh_adjuster = tf.keras.Model(inputs=In, outputs=Out, name="nCh_adjuster")
    
    return mdl_nCh_adjuster

