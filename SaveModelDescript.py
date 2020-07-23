#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from tensorflow.keras.utils import plot_model


# In[ ]:


def SaveModelDescript (model, save_dir=r'./', save_filename='Mdl_Descript'):
    
    if not os.path.exists(os.path.abspath(save_dir)):
        os.makedirs(os.path.abspath(save_dir))
    
    # Open the file
    with open(os.path.join(os.path.abspath(save_dir),str(save_filename+'.txt')),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    plot_model(model, 
               to_file=os.path.join(os.path.abspath(save_dir),str(save_filename+'.png')), 
               show_shapes=True, show_layer_names=True, rankdir="TB", expand_nested=True, dpi=96)
        
    return

