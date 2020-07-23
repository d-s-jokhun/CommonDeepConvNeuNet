#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


def get_CompileParams (model):
    try:
        compile_args = model._get_compile_args()
        optimizer=None; loss=None; metrics=None; loss_weights=None; weighted_metrics=None; run_eagerly=None
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
            run_eagerly = model.run_eagerly
    except:
        optimizer = getattr(model, 'optimizer', None)
        loss = getattr(model, 'loss', None)
        metrics = getattr(model, 'metrics', None)
        loss_weights = getattr(model, 'loss_weights', None)
        weighted_metrics = getattr(model, 'weighted_metrics', None)
        run_eagerly = getattr(model, 'run_eagerly', None)
        print ('compile_args did not work. getattr was used instead.')
        
    CompileParams={'optimizer':optimizer, 'loss':loss, 'metrics':metrics, 'loss_weights':loss_weights, 'weighted_metrics':weighted_metrics, 'run_eagerly':run_eagerly}
    
    return CompileParams

