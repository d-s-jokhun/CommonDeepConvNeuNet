#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from get_LayersOfOrigin import get_LayersOfOrigin
import tensorflow as tf


# In[ ]:


def Reorder_Layers (obj):
#     obj can be a model or a list of layers. The model (or list of layers) has to contain all the layers required for a complete model
    # reorders layers according to dependencies

    if isinstance(obj,tf.keras.models.Model):
        layers = obj.layers
    elif type(obj)==list and all([isinstance(layer,tf.keras.layers.Layer) for layer in obj]):
        layers = obj
    
    # Determining dependencies
    dependencies={}
    for layer in layers:
        try:
            dependents = get_LayersOfOrigin(layer.input)
            if dependents==layer: dependents=[]
            if type(dependents)!=list: dependents=[dependents]
        except:
            dependents = []
        dependencies[str(layer)]= dependents
    
#     one itteration of reordering
    reordered_layers = reorder(layers, dependencies)
    
#     itterative reordering
    while reordered_layers!=layers:
        layers = reordered_layers
        reordered_layers = reorder(layers, dependencies)
    
    return reordered_layers


# In[ ]:


def reorder (layers, dependencies):
# reorders layers according to dependencies
    reordered_layers = []
    # Placing layers without dependency in front
    for layer in layers:
        if dependencies[str(layer)]==[]:
            reordered_layers.append(layer)

    # Handling layers with dependencies
    for layer in layers:
        if layer not in reordered_layers:
            dependents=dependencies[str(layer)]
            for dependent in dependents:
                if dependent not in reordered_layers:
                    reordered_layers.append(dependent)
            reordered_layers.append(layer)
        
    return reordered_layers

