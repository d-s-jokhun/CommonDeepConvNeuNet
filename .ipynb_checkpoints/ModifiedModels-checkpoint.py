#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras


# In[1]:


def nCh_adjuster (NumOfInputCh=1, NumOfOutputCh=3, ImgSize=224):
    
    In = keras.Input(shape=(ImgSize, ImgSize, NumOfInputCh))
    Out = keras.layers.Conv2D(NumOfOutputCh, 1, padding="same", activation="relu", name="ThreeSinglePixFilters")(In)
    nCh_adjuster = keras.Model(inputs=In, outputs=Out, name="nCh_adjuster")
    
    return nCh_adjuster


# In[2]:


# Modified VGG16

def mod_VGG16 (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    VGG16=keras.applications.VGG16(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_VGG16 = keras.Model(inputs=Preprocess.inputs, outputs=VGG16.outputs, name="mod_VGG16")

    mod_VGG16.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )   
    
    return mod_VGG16


# In[3]:


# Modified VGG19

def mod_VGG19 (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    VGG19=keras.applications.VGG19(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_VGG19 = keras.Model(inputs=Preprocess.inputs, outputs=VGG19.outputs, name="mod_VGG19")

    mod_VGG19.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )   
    
    return mod_VGG19


# In[4]:


# Modified ResNet50

def mod_ResNet50 (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    ResNet50=keras.applications.ResNet50(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_ResNet50 = keras.Model(inputs=Preprocess.inputs, outputs=ResNet50.outputs, name="mod_ResNet50")

    mod_ResNet50.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )   
    
    return mod_ResNet50


# In[5]:


# Modified ResNet50V2

def mod_ResNet50V2 (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    ResNet50V2=keras.applications.ResNet50V2(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_ResNet50V2 = keras.Model(inputs=Preprocess.inputs, outputs=ResNet50V2.outputs, name="mod_ResNet50V2")

    mod_ResNet50V2.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )   
    
    return mod_ResNet50V2


# In[6]:


# Modified InceptionV3

def mod_InceptionV3 (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    InceptionV3=keras.applications.InceptionV3(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_InceptionV3 = keras.Model(inputs=Preprocess.inputs, outputs=InceptionV3.outputs, name="mod_InceptionV3")

    mod_InceptionV3.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )   
    
    return mod_InceptionV3


# In[7]:


# Modified Xception

def mod_Xception (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    Xception=keras.applications.Xception(
        include_top=Include_Top,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_Xception = keras.Model(inputs=Preprocess.inputs, outputs=Xception.outputs, name="mod_Xception")

    mod_Xception.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )


    return mod_Xception


# In[8]:


# Modified InceptionResNetV2

def mod_InceptionResNetV2 (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    InceptionResNetV2=keras.applications.InceptionResNetV2(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_InceptionResNetV2 = keras.Model(inputs=Preprocess.inputs, outputs=InceptionResNetV2.outputs, name="mod_InceptionResNetV2")

    mod_InceptionResNetV2.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

   
    
    return mod_InceptionResNetV2


# In[9]:


# Modified NASNetLarge

def mod_NASNetLarge (NumOfClasses, NumOfInputCh=1, ImgSize=224, Weights=None, Include_Top=True):
    
    Preprocess = nCh_adjuster(
        NumOfInputCh=NumOfInputCh, 
        NumOfOutputCh=3, 
        ImgSize=ImgSize
    )
    
    NASNetLarge=keras.applications.NASNetLarge(
        include_top=True,
        weights=Weights,
        input_tensor=Preprocess.outputs[0],
        input_shape=None,
        pooling=None,
        classes=NumOfClasses
    )

    mod_NASNetLarge = keras.Model(inputs=Preprocess.inputs, outputs=NASNetLarge.outputs, name="mod_NASNetLarge")

    mod_NASNetLarge.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

   
    
    return mod_NASNetLarge


# In[ ]:





# In[ ]:




