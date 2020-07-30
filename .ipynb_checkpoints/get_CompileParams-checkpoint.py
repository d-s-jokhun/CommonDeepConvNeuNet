#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


def get_CompileParams (model):
    optimizer = getattr(model, 'optimizer', None)
    loss = getattr(model, 'loss', None)
    metrics = getattr(model, 'metrics', None)
    loss_weights = getattr(model, 'loss_weights', None)
    weighted_metrics = getattr(model, 'weighted_metrics', None)
    run_eagerly = getattr(model, 'run_eagerly', None)
        
    CompileParams={'optimizer':optimizer, 'loss':loss, 'metrics':metrics, 'loss_weights':loss_weights, 'weighted_metrics':weighted_metrics, 'run_eagerly':run_eagerly}
    
    return CompileParams

