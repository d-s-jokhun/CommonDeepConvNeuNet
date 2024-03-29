#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import Lambda_functions


# In[15]:


class mdl_adapter_layers:

    def __init__(self, Output_ImgShape):
        self.Output_ImgShape = Output_ImgShape
                        
    def Im_Flipper(self, name=None):
        return tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", name=name)

    def Im_Rotater(self, name=None):
        rot_layer = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, fill_mode='constant', interpolation='bilinear')
        def rotate (image, rot_layer=rot_layer):
            rotated = tf.numpy_function(rot_layer, [image], tf.float32)
            rotated.set_shape((None,*self.Output_ImgShape))
            return rotated
        return tf.keras.layers.Lambda(rotate, name=name)

    def Im_Resizer(self, name=None):
        return tf.keras.layers.Lambda(lambda image: tf.image.resize_with_pad(
            image, target_height=self.Output_ImgShape[1], target_width=self.Output_ImgShape[0],
            method=tf.image.ResizeMethod.BILINEAR, antialias=True), name=name)
    
    def Int_Rescaler(self, name=None, Saturation=0):
        def rescale (image, Saturation = Saturation):
            rescaled = Lambda_functions.Ch_Norm_255(image, Saturation)
            rescaled.set_shape((None,*self.Output_ImgShape))
            return rescaled
        return tf.keras.layers.Lambda(rescale, name=name)  

    def Ch_Adjuster(self, kernel_regularizer, name=None):
        return tf.keras.layers.Conv2D(filters=self.Output_ImgShape[-1], 
        kernel_size=1, strides=(1, 1), padding="same",data_format=None, 
        activation=None, use_bias=False, kernel_initializer="glorot_uniform", 
        bias_initializer="zeros", kernel_regularizer=kernel_regularizer, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None, 
        name=name)
        


# In[ ]:








# %%
