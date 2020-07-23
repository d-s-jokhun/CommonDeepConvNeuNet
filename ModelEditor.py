#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# (model, New_Layers=None, IncomingLinks_2Axe=None, IncomingLinks_2Forge=None, model_inputs=None, model_outputs=None)
# New_Layers = Layers to add to the model in the form of a dictionary e.g. New_Layers={'drop1':tf.keras.layers.Dropout(rate=0.1),
# 'drop2':tf.keras.layers.Dropout(rate=0.2),}
# IncomingLinks_2Axe = list of integers denoting existing layers in the model whose input links will be severed
# IncomingLinks_2Forge = list of tuples of type (layer in question, layer or list of layers to get input from)
# model_inputs = layer or list of layers or None or []. If none or [], it will be inferred from the model
# model_outputs = layer or list of layers or None or []. If none or [], it will be inferred from the model


# In[ ]:


from get_CompileParams import get_CompileParams
from get_LayersOfOrigin import get_LayersOfOrigin
from reconnect_layers import reconnect_layers
import tensorflow as tf


# In[ ]:


def ModelEditor (model, New_Layers=None, IncomingLinks_2Axe=None, IncomingLinks_2Forge=None, model_inputs=None, model_outputs=None):
        
    if New_Layers!=[] and New_Layers!={} and type(New_Layers)!=type(None):
        assert (type(New_Layers)==dict), 'New_Layers has to be a dictionary!'
        assert all(isinstance(New_Layer, tf.keras.layers.Layer) for New_Layer in New_Layers.values()), 'New_Layers has to be a dictionary containing instances of tf.keras.layers.Layer!'
    
    if type(IncomingLinks_2Forge)==tuple:
        if len(IncomingLinks_2Forge)!=0:
            IncomingLinks_2Forge=[IncomingLinks_2Forge]
    if type(IncomingLinks_2Forge)!=type(None):
        assert all(isinstance(element,tuple) for element in IncomingLinks_2Forge), 'New_inbound_nodes has to be a tuple or a list of tuples!'
    
#     Getting the parameters required for compiling the model later
    optimizer,loss,metrics,loss_weights,weighted_metrics,run_eagerly = get_CompileParams (model).values()

#     Getting model input and output layers
    if model_inputs==[] or model_inputs==None:
        model_inputs=model.inputs
    if model_outputs==[] or model_outputs==None:
        model_outputs=model.outputs
    output_layers = get_LayersOfOrigin(model_outputs)
    if type(output_layers)!=list: output_layers=[output_layers]
    input_layers = get_LayersOfOrigin(model_inputs) 
    if type(input_layers)!=list: input_layers=[input_layers]

    
#     Editing the model

# Axing links
    if IncomingLinks_2Axe!=[] and type(IncomingLinks_2Axe)!=type(None):
        if type(IncomingLinks_2Axe)!=list and type(IncomingLinks_2Axe)!=tuple:
            IncomingLinks_2Axe=[IncomingLinks_2Axe]
        for IncomingLink_2Axe in IncomingLinks_2Axe:
            model.layers[IncomingLink_2Axe].inbound_nodes.clear()

            
# Forging new links
    if IncomingLinks_2Forge!=[] and type(IncomingLinks_2Forge)!=type(None):
        for IncomingLink_2Forge in IncomingLinks_2Forge:
            if type(IncomingLink_2Forge[1])!=list:
                IncomingLink_2Forge[0].__call__(IncomingLink_2Forge[1].output)
            else:
                IncomingLink_2Forge[0].__call__([IncomingLink_2Forge[1][n].output for n in range(len(IncomingLink_2Forge[1]))])

            
# Inserting the new layers in the model            
    if type(New_Layers)!=type(None):
        if len(New_Layers)>0:
            for New_Layer_Key in list(New_Layers.keys()):
                model._insert_layers(New_Layers[New_Layer_Key])
                
                
    reconnect_layers(model, input_layers=input_layers, reorder_layers=True)
                
      
            
#     Building the edited model
    model = tf.keras.Model(inputs=[input_layer.input for input_layer in input_layers], 
                           outputs=[output_layer.output for output_layer in output_layers], 
                           name=str(model.name+'_edited')) 
    
#     Compiling the model using parameters obtained before reloading the updated model
    if optimizer!=None:
        model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics,
            loss_weights = loss_weights,
            weighted_metrics = weighted_metrics,
            run_eagerly = run_eagerly
        )
        print ('Model successfully edited and recompiled!')
    else:
        print ('Model successfully edited!')
    
    return model

