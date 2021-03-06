#!/usr/bin/env python
# coding: utf-8

# #### Add L1L2 regularization to an existing model

# In[ ]:


import random
import os
import tensorflow as tf
from get_CompileParams import get_CompileParams


# In[3]:


def RegularizeModel(model, regularizer, keep_weights=True):
    
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model    
    
#     Adding regularization if the layer is regularizable
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            setattr(layer, 'kernel_regularizer', regularizer)

    # When we change the layer attributes, the change only happens in the model config file
    if keep_weights:
        weights = model.get_weights() # Get the weights before reloading the model
        model = tf.keras.models.Model.from_config(model.get_config()) # recreates the model from the updated config file
        model.set_weights(weights) # Reload the model weights
    else:
        model = tf.keras.models.Model.from_config(model.get_config()) # recreates the model from the updated config file
    
       
    return model


# In[ ]:




