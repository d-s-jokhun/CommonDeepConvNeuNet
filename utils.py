#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import math


# In[3]:



def ImportImage (path, ScaleFactor=1, ImgSize=0):
       
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
    W = math.floor(img.shape[1] * ScaleFactor)
    H = math.floor(img.shape[0] * ScaleFactor)
    
    if ImgSize>0 and (W>ImgSize or H>ImgSize):
        return
    
    img = cv2.resize(img, (W,H))
    img = np.subtract(img, np.amin(img))
    img = np.divide(img, np.amax(img))

    
    width_to_pad=ImgSize-img.shape[1]
    height_to_pad=ImgSize-img.shape[0]
    
    width_start, width_end, height_start, height_end = (0,0,0,0)
    
    if width_to_pad>0:
        width_start = width_to_pad//2
        width_end = width_to_pad - width_start
        
    if height_to_pad>0:
        height_start = height_to_pad//2
        height_end = height_to_pad - height_start
        
    if len(img.shape) == 2:
        img = np.pad(img,((height_start,height_end),(width_start,width_end)))
    elif len(img.shape) == 3:
        img = np.pad(img,((height_start,height_end),(width_start,width_end),(0,0)))
    
    return img


# In[41]:



def DataPartition(FullSet, Partition=[0.6,0.2,0.2], RanSeed=None):
    assert np.sum(Partition)<=1
    np.random.seed(RanSeed)
    l = len(FullSet)
    new_order = np.random.choice(FullSet, size = l, replace = False)

    Tr_MaxIdx = np.ceil(l*Partition[0]).astype(np.int)
    Dev_MaxIdx = np.ceil(l*(Partition[0]+Partition[1])).astype(np.int)
    Ts_MaxIdx = np.ceil(l*(Partition[0]+Partition[1]+Partition[2])).astype(np.int)

    Tr_Set = new_order[0:Tr_MaxIdx]
    Dev_Set = new_order[Tr_MaxIdx:Dev_MaxIdx]
    Ts_Set = new_order[Dev_MaxIdx:Ts_MaxIdx]
    
    return Tr_Set, Dev_Set, Ts_Set


# In[ ]:


# In[ ]:





# In[ ]:




