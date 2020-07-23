#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


def get_LayersOfOrigin (objs):
# objs can be a single or a list of layer/s and/or tensor/s

    if type(objs)==list or type(objs)==tuple:
        layers = []
        for obj in objs:
            if isinstance(obj,tf.keras.layers.Layer):
                layers.append(obj)
            else:
                layers.append(obj.__dict__['_keras_history'].layer)
    else:
        if isinstance(objs,tf.keras.layers.Layer):
            layers = objs
        else:
            layers = objs.__dict__['_keras_history'].layer  
            
    return layers

