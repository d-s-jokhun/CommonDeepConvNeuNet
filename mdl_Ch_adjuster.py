#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import Lambda_functions


# In[15]:


def mdl_Ch_adjuster (Input_ImgSize, Output_ImgSize):
        
    assert (len(Output_ImgSize)==3 and len(Output_ImgSize)==3), 'Input_ImgSize and Output_ImgSize each has to be a tuple or a list of 3 elements (height, width, num of channels)'
        
    In = tf.keras.Input(shape=Input_ImgSize, name="Input")

    x = tf.keras.layers.Lambda(
        lambda image: tf.image.resize_with_pad(
            image, target_height=Output_ImgSize[0], target_width=Output_ImgSize[1], 
            method=tf.image.ResizeMethod.BILINEAR, antialias=True), name="Resizing")(In)
    
    x = tf.keras.layers.Conv2D(
        filters=Output_ImgSize[2], kernel_size=1, strides=(1, 1), padding="same",data_format=None, 
        activation=None, use_bias=False, kernel_initializer="glorot_uniform", bias_initializer="zeros",
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, bias_constraint=None, name="SinglePixConv")(x) 
    
    Out=tf.keras.layers.Lambda(
        Lambda_functions.Ch_Norm_255, name='Ch_Norm_255')(x)
    
    mdl_Ch_adjuster = tf.keras.Model(inputs=In, outputs=Out, name="Ch_adjuster")
    
    return mdl_Ch_adjuster


# In[ ]:




