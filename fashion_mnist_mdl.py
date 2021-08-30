#!/usr/bin/env python
# coding: utf-8

#%%


#%%

# %load_ext autoreload
# %autoreload 2

import os
from os import listdir
from os.path import join, basename
import numpy as np
import tifffile
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from mdl_adapter_layers import mdl_adapter_layers
from SaveModelDescript import SaveModelDescript
from change_InputShape import change_InputShape
from callback_PredWriter import callback_PredWriter
from datetime import datetime
import time
import cv2

# import matplotlib
# from RegularizeModel import RegularizeModel
# from ModelEditor import ModelEditor
# from get_CompileParams import get_CompileParams
# from callback_ConfMat import callback_ConfMat
# import pathlib
# import pandas as pd
# from Dataset_from_Dataframe import Dataset_from_Dataframe
# from Dataset_from_Cache import Dataset_from_Cache
# import pickle

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


#%% Defining paths, loading dataframes and identifying classes

# MasterPath = pathlib.Path("/gpfs0/home/jokhun/Pro 1/U2OS small mol screening")
# MasterPath = os.path.abspath('//deptnas.nus.edu.sg/BIE/MBELab/jokhun/fashion_mnist')
# MasterPath = os.path.abspath('/gpfs0/home/jokhun/Pro 1/Cellular biosensor')
MasterPath = os.path.abspath('/MBELab/jokhun/fashion_mnist')

model_name = 'fashion_Trans'

ImgPath = join (MasterPath,'data_raw')

Ch1_img_paths = []
Ch2_img_paths = []
Ch3_img_paths = []
Ch4_img_paths = []
Ch5_img_paths = []
for (root,dirs,files) in os.walk(ImgPath):
    if len(files)>0:
        for filename in files:
            if '.tif' in filename:
                Ch1_img_paths.append(join(root,filename))

# file_list = listdir(ImgPath)
# Ch1_img_paths = []
# Ch2_img_paths = []
# Ch3_img_paths = []
# Ch4_img_paths = []
# Ch5_img_paths = []
# for file_name in file_list:
#     if 'DIC' in file_name: 
#         Ch1_img_paths.append(join(ImgPath,file_name))
#     # if 'Lysosome' in file_name: 
#     #     Ch2_img_paths.append(join(ImgPath,file_name))
#     # if 'ER' in file_name: 
#     #     Ch3_img_paths.append(join(ImgPath,file_name))
#     # if 'MT' in file_name: 
#     #     Ch4_img_paths.append(join(ImgPath,file_name))
#     # if 'Nuc' in file_name: 
#     #     Ch5_img_paths.append(join(ImgPath,file_name))

def prepare_paths (paths,order):
    if len(paths)>1:
        paths.sort()
        def sort_key(item):
            return item[-1]
        paths=list(zip(paths,order))
        paths.sort(key=sort_key)
        paths=list(list(zip(*paths))[0])
    return paths

np.random.seed(0)
order = np.random.choice(list(range(len(Ch1_img_paths))), size=len(Ch1_img_paths), replace = False)
Ch1_img_paths = prepare_paths(Ch1_img_paths,order)
Ch2_img_paths = prepare_paths(Ch2_img_paths,order)
Ch3_img_paths = prepare_paths(Ch3_img_paths,order)
Ch4_img_paths = prepare_paths(Ch4_img_paths,order)
Ch5_img_paths = prepare_paths(Ch5_img_paths,order)


# sample_id = np.random.choice([i for i in range(len(Ch1_img_paths))], size=1000, replace = False)
# Ch1_img_paths = [Ch1_img_paths[i] for i in sample_id]
# Ch2_img_paths = [Ch2_img_paths[i] for i in sample_id]
# Ch3_img_paths = [Ch3_img_paths[i] for i in sample_id]
# Ch4_img_paths = [Ch4_img_paths[i] for i in sample_id]
# Ch5_img_paths = [Ch5_img_paths[i] for i in sample_id]

#%% Function to load images

def load_img (path):
    # return tifffile.imread(path)
    return cv2.resize(tifffile.imread(path), (128,128), interpolation = cv2.INTER_AREA)

#%% Loading images

if __name__=='__main__':
    with mp.Pool() as pool:
        Ch1_img = np.array(pool.map(load_img,Ch1_img_paths))
        Ch2_img = np.array(pool.map(load_img,Ch2_img_paths))
        Ch3_img = np.array(pool.map(load_img,Ch3_img_paths))
        Ch4_img = np.array(pool.map(load_img,Ch4_img_paths))
        Ch5_img = np.array(pool.map(load_img,Ch5_img_paths))

#%%

X = np.expand_dims(Ch1_img,-1)
# X = np.stack((Ch1_img,Ch2_img,Ch3_img,Ch4_img,Ch5_img),axis=3)
First_ClassDelimiter = '_'
Second_ClassDelimiter = '_'
Y = [basename(path) for path in Ch1_img_paths]
Y = [y[y.find(First_ClassDelimiter)+1:] for y in Y]
Y = [y[:y.find(Second_ClassDelimiter)] for y in Y]

Ts_size = int(0.0 * X.shape[0])
Val_size = int(0.15 * X.shape[0])
Tr_size = X.shape[0] - (Ts_size + Val_size)
print(f'Train, val and test sample sizes: {Tr_size}, {Val_size}, {Ts_size}')

Ts_X = X[0:Ts_size]
Val_X = X[Ts_size:Ts_size + Val_size]
Tr_X = X[Ts_size + Val_size:Ts_size + Val_size + Tr_size]

Ts_img_paths = Ch1_img_paths[0:Ts_size]
Val_img_paths = Ch1_img_paths[Ts_size:Ts_size + Val_size]
Tr_img_paths = Ch1_img_paths[Ts_size + Val_size:Ts_size + Val_size + Tr_size]

#%% Encoding labels

ResponseEncoder = LabelEncoder()
ResponseEncoder.fit(Y)

class_names = ResponseEncoder.classes_
NumOfClasses = len(class_names)
print('Number of classes in the data: '+str(NumOfClasses))
print(class_names)

Ts_Y = ResponseEncoder.transform(Y[0:Ts_size])
Val_Y = ResponseEncoder.transform(Y[Ts_size:Ts_size + Val_size])
Tr_Y = ResponseEncoder.transform(Y[Ts_size + Val_size:Ts_size + Val_size + Tr_size])


#%%

batch_size = 128
shuffle_buffer_size = np.max([batch_size,NumOfClasses])*2

def idx2img(batch_idx,batch_y,ImageSet):
    def fetch (batch_idx,ImageSet=ImageSet):
        batch_x = ImageSet[batch_idx]
        return batch_x
    batch_x=tf.numpy_function(fetch, [batch_idx], tf.uint8)
    batch_x.set_shape((None,*ImageSet.shape[1:]))
    return batch_x,batch_y
def idx2img_Tr(batch_idx,batch_y,ImageSet=Tr_X):
    return idx2img(batch_idx,batch_y,ImageSet)
def idx2img_Val(batch_idx,batch_y,ImageSet=Val_X):
    return idx2img(batch_idx,batch_y,ImageSet)
def idx2img_Ts(batch_idx,batch_y,ImageSet=Ts_X):
    return idx2img(batch_idx,batch_y,ImageSet)

dataset_Tr = tf.data.Dataset.from_tensor_slices((np.array(range(len(Tr_X))), Tr_Y)).shuffle(buffer_size=shuffle_buffer_size)\
    .batch(batch_size=batch_size).map(idx2img_Tr, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)\
        .prefetch(tf.data.experimental.AUTOTUNE)

dataset_Val = tf.data.Dataset.from_tensor_slices((np.array(range(len(Val_X))), Val_Y)).shuffle(buffer_size=shuffle_buffer_size)\
    .batch(batch_size=batch_size).map(idx2img_Val, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)\
        .prefetch(tf.data.experimental.AUTOTUNE)

dataset_Ts = tf.data.Dataset.from_tensor_slices((np.array(range(len(Ts_X))), Ts_Y)).shuffle(buffer_size=shuffle_buffer_size)\
    .batch(batch_size=batch_size).map(idx2img_Ts, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)\
        .prefetch(tf.data.experimental.AUTOTUNE)


#%% function to change the input size of the core model

def change_InputShape (model, new_InputShape):

    model._layers[0]._batch_input_shape = (None,*new_InputShape)

    # rebuild model architecture by exporting and importing via json
    new_model = tf.keras.models.model_from_json(model.to_json())

    # copy weights from old model to new one
    success=[]
    failure=[]
    for l,layer in enumerate(new_model.layers):
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            success.append((l,layer.name))
        except:
            failure.append((l,layer.name))
    print(f'Weights successfully transferred to {len(success)} layers')
    print(f'Failed to transfer weights to {len(failure)} layers')
    print(failure)

    return new_model

#%% Building the model

ModelName = model_name+'_Xception'
core_model = tf.keras.applications.Xception(include_top=False, weights="imagenet")
ref_core_model = "tf.keras.applications.Xception(include_top=False, weights=None)"
core_preprocess_layers = tf.keras.applications.xception.preprocess_input


AddTopAndBottomLayers = True

regularizer_Bottom = tf.keras.regularizers.l1_l2(l1=0, l2=0.001)
original_img_shape = dataset_Tr.element_spec[0].shape.as_list()[1:]  #Tr_X[0].shape #(w, h, No._of_Ch)
Final_img_shape = (original_img_shape)
if AddTopAndBottomLayers:

    core_model = change_InputShape (core_model,Final_img_shape)

    In = tf.keras.Input(shape=original_img_shape, name="Preprocess_Input")
    adapter_layers = mdl_adapter_layers (Output_ImgShape=Final_img_shape)
    Out = core_preprocess_layers(adapter_layers.Int_Rescaler(name='Ch_rescale_0_255', Saturation=[0.001,0.005])(\
            adapter_layers.Im_Rotater(name='random_rotate')(\
                adapter_layers.Im_Flipper(name='random_flip_HnV')(\
                    adapter_layers.Im_Resizer(name='resize')(In)))))

    mdl_preprocess = tf.keras.Model(inputs=In, outputs=Out, name='mdl_preprocess')

    
    In = tf.keras.Input(shape=core_model.output_shape[1:4], name="Features")
    GlbMaxPool = tf.keras.layers.GlobalMaxPool2D(name="GlbMaxPool")
    GlbAvgPool = tf.keras.layers.GlobalAveragePooling2D(name="GlbAvgPool")
    MaxAvgConcat = tf.keras.layers.Concatenate(axis=-1, name="MaxAvgConcat")
    Dropout1 = tf.keras.layers.Dropout(0.5, name="Feature_Dropout")
    Dense1 = tf.keras.layers.Dense(units=core_model.output_shape[-1], activation='relu', kernel_regularizer=regularizer_Bottom, name="dense1")
    Dropout2 = tf.keras.layers.Dropout(0.5, name="Dropout2")
    predictions = tf.keras.layers.Dense(units=NumOfClasses, activation=None, kernel_regularizer=regularizer_Bottom, name="predictions")
    Out=predictions(Dropout2(Dense1(Dropout1(MaxAvgConcat([GlbMaxPool(In),GlbAvgPool(In)])))))
    mdl_prediction = tf.keras.Model(inputs=In, outputs=Out, name='mdl_prediction')

    In = tf.keras.Input(shape=original_img_shape, name="Input_images")
    Out = mdl_prediction(core_model(mdl_preprocess(In)))
    model = tf.keras.Model(inputs=In, outputs=Out, name=ModelName)
    print ('Bottom and Top layers added!')    


#%% Saving model description

SaveModelDescription = True

if SaveModelDescription:
    Model_Path = join(MasterPath,model.name)    
    SaveModelDescript(model, save_dir=Model_Path,\
        save_filename=model.name)        
    print ('Model descriptions saved!')


#%% Editing the core model

# Edit_Core_Model = False
# # SaveEditedModelDescription = True
# # ModelKey = ModelKeys[0]

# if Edit_Core_Model:
#     model = models[ModelKey]
#     New_Layers={'drop1':tf.keras.layers.Dropout(rate=0.1, name='drop1'),
#                 'drop2':tf.keras.layers.Dropout(rate=0.2, name='drop2'),
#                 'drop3':tf.keras.layers.Dropout(rate=0.3, name='drop3'),
#                 'drop4':tf.keras.layers.Dropout(rate=0.4, name='drop4'),
#                 'drop5':tf.keras.layers.Dropout(rate=0.5, name='drop5'),
#                }

#     IncomingLinks_2Axe=[-18, -12, -14, -8, -5, -1]   

#     IncomingLinks_2Forge=[(New_Layers['drop1'], model.layers[-19]),
#                           (model.layers[-18], New_Layers['drop1']),
#                           (model.layers[-12], New_Layers['drop1']),
#                           (New_Layers['drop2'], model.layers[-15]),
#                           (model.layers[-14], New_Layers['drop2']),
#                           (New_Layers['drop3'], model.layers[-9]),
#                           (model.layers[-8], New_Layers['drop3']),
#                           (New_Layers['drop4'], model.layers[-6]),
#                           (model.layers[-5], New_Layers['drop4']),
#                           (New_Layers['drop5'], model.layers[-2]),
#                           (model.layers[-1], New_Layers['drop5']),
#                          ]

#     model_inputs=None
#     model_outputs=None

#     model = ModelEditor(model, New_Layers=New_Layers, IncomingLinks_2Axe=IncomingLinks_2Axe, 
#                                 IncomingLinks_2Forge=IncomingLinks_2Forge,
#                                 model_inputs=model_inputs, model_outputs=model_outputs)
#     models[ModelKey] = model 
    
#     # Save edited model description
#     if SaveEditedModelDescription:
#         Model_Path = os.path.join(MasterPath,str(ModelKey))        
#         SaveModelDescript(model, save_dir=Model_Path, 
#                           save_filename=str(ModelKey+'_edited'))        
#         print ('Model descriptions saved!')


#%% Adding Regularization to all regularizable layers

# RegularizeTheModel = False
# if RegularizeTheModel:
#     regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.001)
#     for ModelKey in ModelKeys:
#         models[ModelKey]=RegularizeModel(models[ModelKey], regularizer, keep_weights=True)
# else: print ('No alteration in regularization!')


#%% Summary of models

determine_initial_accuracies = False

TrainEval=[]
ValEval=[]
TestEval=[]
if determine_initial_accuracies:

    model.compile(optimizer='adam',\
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
            metrics=['accuracy'])
    print ('Model compiled!')

    print ('Initial Train Loss and Accuracy')
    Eval=model.evaluate(dataset_Tr, verbose=0)
    TrainEval.append(Eval)
    print (TrainEval) 

    print ('\nInitial Val Loss and Accuracy')
    Eval=model.evaluate(dataset_Val, verbose=0)
    ValEval.append(Eval)
    print (ValEval)

    print ('\nInitial Test Loss and Accuracy')
    Eval=model.evaluate(dataset_Ts, verbose=0)
    TestEval.append(Eval)
    print (TestEval)


#%% Training config

def train_model (model, Dataset_Tr, Dataset_Val, initial_epoch=0, final_epoch=5, Model_Path=None, class_names=None):
    if Model_Path==None or Model_Path==[]:
        Model_Path=model.name        
        
    sess_DateTime = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
    MdlChkpt_Path = os.path.join(Model_Path,"MdlChkpt",(sess_DateTime+"_e{epoch:03d}_Acc{accuracy:.2f}_ValAcc{val_accuracy:.2f}.ckpt"))
    MdlChkpt_cb = tf.keras.callbacks.ModelCheckpoint(
        MdlChkpt_Path, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True, 
        mode='auto', save_freq="epoch"
    )
    TensorBoard_Path = os.path.join(Model_Path,"logs",(model.name+'_'+sess_DateTime))
    TensorBoard_cb = tf.keras.callbacks.TensorBoard(
        log_dir = TensorBoard_Path, histogram_freq=0, write_graph=False, write_images=False, update_freq="epoch", 
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )

    EarlyStop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
        min_delta=0, patience=150, verbose=2, mode='auto',\
            baseline=None, restore_best_weights=True)
    
    # ConfMat_Path = os.path.join(Model_Path,"logs",(model.name+'_'+sess_DateTime))
    # log_confusion_matrix=callback_ConfMat(model, Dataset_Val, class_names=class_names, logdir=ConfMat_Path, freq=10)
    # # Define the per-epoch callback.
    # ConfMat_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    PredLog_Path = os.path.join(Model_Path,"logs",(model.name+'_'+sess_DateTime))
    pred_writer = callback_PredWriter(model, Dataset_Val, class_names=class_names, logdir=PredLog_Path, freq=5) #freq in epochs
    # Define the per-epoch callback.
    PredWriter_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=pred_writer)
        
    history = model.fit(Dataset_Tr, initial_epoch=initial_epoch, epochs=final_epoch, \
        verbose=1, callbacks=[EarlyStop_cb, TensorBoard_cb, MdlChkpt_cb, PredWriter_cb], # ,ConfMat_cb
        validation_data=Dataset_Val, validation_freq=1)        
    
    return history


#%%

FreshTrainHistory = True
if FreshTrainHistory:
    history={}


#%% Selecting the layers to be trained

FineTune_Core_Onwards = None  # 0 to set the entire model as trainable and None to copletely freeze training of the core model
learning_rate = 1e-3 # 1e-3, 1e-7
CoreModel_layer = -2

Change_Classifier_DropoutRate = False
New_DropoutRate = 0.8

print ('Total params:',f'{np.sum([np.prod(a.shape) for a in model.weights]):,}')
print ('Trainable params:',f'{np.sum([np.prod(a.shape) for a in model.trainable_weights]):,}')
print ('Non-trainable params:',f'{np.sum([np.prod(a.shape) for a in model.non_trainable_weights]):,}')
core_model = model.layers[CoreModel_layer]
core_model.trainable = True
if FineTune_Core_Onwards == None:
    FineTune_Core_Onwards = len(core_model.layers)
Num_CoreTrainableLayers=0
i_track=-1
for layer in core_model.layers[:FineTune_Core_Onwards]:
    i_track+=1
    if i_track > 1:
        layer.trainable = False
if FineTune_Core_Onwards != len(core_model.layers):
    ref_mdl = eval(ref_core_model)
    for layer in core_model.layers[FineTune_Core_Onwards:]:
        i_track+=1
        layer.trainable = ref_mdl.layers[i_track].trainable
        Num_CoreTrainableLayers+=1
    del ref_mdl
print ('Appropriate layers after',FineTune_Core_Onwards,'of',core_model.name,'have been set as trainable!')

if Change_Classifier_DropoutRate:
    model.layers[-1].layers[-2].rate = New_DropoutRate
    model.layers[-1].layers[-4].rate = New_DropoutRate
    print ('The dropout rates of the dropout layers in the classifier has been altered to', New_DropoutRate)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),\
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
        metrics=['accuracy'])
print (model.name,'has been recompiled!')
print ('Total params:',f'{np.sum([np.prod(a.shape) for a in model.weights]):,}')
print ('Trainable params:',f'{np.sum([np.prod(a.shape) for a in model.trainable_weights]):,}')
print ('Non-trainable params:',f'{np.sum([np.prod(a.shape) for a in model.non_trainable_weights]):,}')




#%% Training

Epochs2TrainFor= 1000

his_log=sorted(list(history.keys()))
if len(his_log)>0:
    his_log = history[his_log[-1]]
    initial_epoch = his_log.epoch[-1] + 1
else: initial_epoch = 0
final_epoch = initial_epoch + Epochs2TrainFor

ModelStart=time.perf_counter()
print('\nTraining!')
Model_Path = os.path.join(MasterPath,model.name)
sess_DateTime = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
history[model.name+'_'+sess_DateTime+'_'+str(Num_CoreTrainableLayers)]=train_model (
    model, dataset_Tr, dataset_Val, initial_epoch=initial_epoch, final_epoch=final_epoch,
    Model_Path=Model_Path, class_names=class_names)

print('\n'+model.name+' trained! Training time = '+ str((time.perf_counter()-ModelStart)/60) + ' min!')


#%% Summary of Models

determine_accuracies = False

if determine_accuracies:
    print ('Train Loss and Accuracy')
    Eval=model.evaluate(dataset_Tr, verbose=0)
    TrainEval.append(Eval)
    for Eval in TrainEval:
        print (Eval) 

    print ('\nVal Loss and Accuracy')
    Eval=model.evaluate(dataset_Val, verbose=0)
    ValEval.append(Eval)
    for Eval in ValEval:
        print (Eval)

    print ('\nTest Loss and Accuracy')
    Eval=model.evaluate(dataset_Ts, verbose=0)
    TestEval.append(Eval)
    for Eval in TestEval:
        print (Eval)


#%% Saving the latest version of the model

SaveLatestVersions = True
if SaveLatestVersions:

    In = tf.keras.Input(shape=Final_img_shape, name="core_input_saving")
    Out = mdl_prediction(core_model(In))
    model_saving = tf.keras.Model(inputs=In, outputs=Out, name='model_saving')
    sess = datetime.now().strftime("%Y%m%d-%H%M%S")
    Save_Path = os.path.join(MasterPath,model.name,("LastModel_"+sess))
    model_saving.save(Save_Path)
    # tf.keras.models.save_model(models[ModelKey], Save_Path, \
    #     overwrite=False, include_optimizer=True)
    # Save_Path = os.path.join(MasterPath,str(ModelKey),("history_"+sess+'.pkl'))
    # with open(Save_Path,"wb") as history_file:
    #     pickle.dump(history,history_file)
    print('\nThe latest version of the model has been saved!')


#%%




