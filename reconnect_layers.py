#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Reorder_Layers import Reorder_Layers
from get_LayersOfOrigin import get_LayersOfOrigin
import tensorflow as tf


# In[ ]:


def reconnect_layers (obj, input_layers=None, reorder_layers=True):
#     obj can be a model or a list of layers. The model (or list of layers) has to contain all the layers required for a complete model
# reconnects layers according to updated tensors
# If reorder_layers=True, reordering is performed according to dependencies the layers
    if input_layers==None or input_layers==[]:
        assert isinstance(obj,tf.keras.models.Model), 'If input_layers are not provided, obj has to be a model (instance of tf.keras.models.Model!)'
        input_layers = get_LayersOfOrigin(obj.inputs)
        if type(input_layers)!=list and type(input_layers)!=tuple:
            input_layers = [input_layers]     
    else:
        if type(input_layers)!=list and type(input_layers)!=tuple:
            input_layers = [input_layers]        
        assert all(isinstance(input_layer,tf.keras.layers.Layer) for input_layer in input_layers), 'Input layer/s has/have to be instance/s of tf.keras.layers.Layer!)'

#         Obtaining the layers
    if isinstance(obj,tf.keras.models.Model):
        layers = obj.layers
    elif type(obj)==list and all([isinstance(layer,tf.keras.layers.Layer) for layer in obj]):
        layers = obj
    
        
#     Reordering layers according to dependencies
    if reorder_layers:
        layers = Reorder_Layers(layers)
        
    # Reconnecting all layers to updated incoming tensors
    for layer in layers:
        if layer not in input_layers:
            try:
                updated_IncomingTensors = get_LayersOfOrigin(layer.input)
                if type(updated_IncomingTensors)==list:
                    updated_IncomingTensors = [L.output for L in updated_IncomingTensors]
                else:
                    updated_IncomingTensors = updated_IncomingTensors.output
                layer.inbound_nodes.clear()
                layer.__call__(updated_IncomingTensors)
            except:
                pass

    return

