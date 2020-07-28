#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[1]:


def mdl_nCh_adjuster (NumOfInputCh=1, NumOfOutputCh=3, ImgSize=(224,224)):
    
    In = tf.keras.Input(shape=(ImgSize[0], ImgSize[1], NumOfInputCh), name="Input")
    Out = tf.keras.layers.Conv2D(NumOfOutputCh, 1, padding="same", activation="relu", name="ThreeSinglePixFilters")(In)
    mdl_nCh_adjuster = tf.keras.Model(inputs=In, outputs=Out, name="nCh_adjuster")
    
    return mdl_nCh_adjuster

