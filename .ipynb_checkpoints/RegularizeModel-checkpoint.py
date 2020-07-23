#!/usr/bin/env python
# coding: utf-8

# #### Add L1L2 regularization to an existing model

# In[ ]:


import random
import os
import tensorflow as tf


# In[3]:


def RegularizeModel(model, regularizer, keep_weights=True):
    
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model    
    
#     Getting the parameters required for compiling the model later
    compile_args = model._get_compile_args()
    optimizer="adam"; loss=None; metrics=None; loss_weights=None; weighted_metrics=None; run_eagerly=None
    if hasattr(model, 'optimizer'):
        optimizer = compile_args['optimizer']
    if hasattr(model, 'loss'):
        loss = compile_args['loss']
    if hasattr(model, 'metrics'):
        metrics = compile_args['metrics']
    if hasattr(model, 'loss_weights'):
        loss_weights = compile_args['loss_weights']
    if hasattr(model, 'weighted_metrics'):
        weighted_metrics = compile_args['weighted_metrics']
    if hasattr(model, 'run_eagerly'):
        run_eagerly = compile_args['run_eagerly']
    
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
    
#     Compiling the model using parameters obtained before reloading the updated model
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics,
        loss_weights = loss_weights,
        weighted_metrics = weighted_metrics,
        run_eagerly = run_eagerly
    )
       
    return model


# In[ ]:




