#!/usr/bin/env python
# coding: utf-8

# In[1]:


# readme

# import datetime
# import os
# import tensorflow as tf

# sess_DateTime = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
# ConfMat_Path = os.path.join(Model_Path,"logs",(model.name+'_'+sess_DateTime))
# log_confusion_matrix=callback_ConfMat(model=model, X=X_Val, Y=Y_Val, class_names=class_names, logdir=ConfMat_Path)
# Define the per-epoch callback.
# ConfMat_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# add ConfMat_cb in the list of callbacks while training


# In[5]:

import numpy as np
import os
from csv import writer
import tensorflow as tf
import pathlib


#%%

def callback_PredWriter(model, dataset, class_names, logdir, freq=1):
    
    if type(freq)==type(None) or freq==[]:
        freq=1

    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(logdir,'pred.csv')

    if not os.path.isfile(file_path):
        Y_callback_PredWriter=[]
        for batch in dataset:
            _,lbl = batch
            Y_callback_PredWriter.extend(lbl.numpy())

        with open(file_path, 'w', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(['class_names']+list(class_names))        
            csv_writer.writerow(['ground_truth']+Y_callback_PredWriter)

    def log_predictions(epoch, logs):
        if epoch==0 or (epoch+1)%freq == 0:
            # Use the model to predict the values from the validation dataset.
            pred = model.predict(dataset)
            pred = np.argmax(pred, axis=1)
            
            with open(file_path, 'a', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow([f'epoch: {epoch}']+list(pred))  

    return log_predictions


# In[ ]:





# In[ ]:




