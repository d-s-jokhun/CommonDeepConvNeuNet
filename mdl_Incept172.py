#!/usr/bin/env python
# coding: utf-8

#%%

# %load_ext autoreload
# %autoreload 2

import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from mdl_adapter_layers import mdl_adapter_layers
import os
from RegularizeModel import RegularizeModel
from SaveModelDescript import SaveModelDescript
from ModelEditor import ModelEditor
from get_CompileParams import get_CompileParams
from callback_ConfMat import callback_ConfMat
from callback_PredWriter import callback_PredWriter
import pathlib
import pandas as pd
from Dataset_from_Dataframe import Dataset_from_Dataframe
from Dataset_from_Cache import Dataset_from_Cache
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#%% Defining paths, loading dataframes and identifying classes

MasterPath = pathlib.Path("/gpfs0/home/jokhun/Pro 1/U2OS small mol screening")
# MasterPath = os.path.abspath('//deptnas.nus.edu.sg/BIE/MBELab/jokhun/Pro 1/U2OS small mol screening')

original_img_shape = (172,172,5)
Img_dir = str(MasterPath.joinpath('im_BigFields(172,172)_5Ch'))
Dataset_dir = MasterPath.joinpath('Datasets_BigFields_5Ch')
DataFrame_Tr_path = Dataset_dir.joinpath('DF_10cls_(172,172)_Tr.csv.xz')
DataFrame_Val_path = Dataset_dir.joinpath('DF_10cls_(172,172)_Val.csv.xz')
DataFrame_Ts_path = Dataset_dir.joinpath('DF_10cls_(172,172)_Ts.csv.xz')

DataFrame_Tr = pd.read_csv(DataFrame_Tr_path)
DataFrame_Val = pd.read_csv(DataFrame_Val_path)
DataFrame_Ts = pd.read_csv(DataFrame_Ts_path)

Classes_Tr = pd.DataFrame([os.path.dirname(Class)[2:-4] for Class in DataFrame_Tr['rel_path']],columns=['Classes'])
Classes_Val = pd.DataFrame([os.path.dirname(Class)[2:-4] for Class in DataFrame_Val['rel_path']],columns=['Classes'])
Classes_Ts = pd.DataFrame([os.path.dirname(Class)[2:-4] for Class in DataFrame_Ts['rel_path']],columns=['Classes'])

print ('Training, validation and test dataframes loaded!')
print (f'Train, Val and Test dataset sizes:\n {DataFrame_Tr.shape[0]}, {DataFrame_Val.shape[0]}, {DataFrame_Ts.shape[0]}')


#%% Encoding labels and adding it to the respective dataframes

ResponseEncoder = LabelEncoder()
ResponseEncoder.fit(Classes_Tr['Classes'].append(Classes_Val['Classes']).append(Classes_Ts['Classes']))

class_names = ResponseEncoder.classes_
NumOfClasses = len(class_names)
print('Number of classes in the data: '+str(NumOfClasses))

DataFrame_Tr = pd.concat([DataFrame_Tr['rel_path'],\
    pd.DataFrame(ResponseEncoder.transform(Classes_Tr['Classes']),columns=['label'])\
        ],axis=1) 
DataFrame_Val = pd.concat([DataFrame_Val['rel_path'],\
    pd.DataFrame(ResponseEncoder.transform(Classes_Val['Classes']),columns=['label'])\
        ],axis=1) 
DataFrame_Ts = pd.concat([DataFrame_Ts['rel_path'],\
    pd.DataFrame(ResponseEncoder.transform(Classes_Ts['Classes']),columns=['label'])\
        ],axis=1) 

print('\n1st rel_path of DataFrame_Tr:',DataFrame_Tr['rel_path'][0])
print('1st Class of Training dataframe:',Classes_Tr['Classes'][0])
print('1st label of DataFrame_Tr:',DataFrame_Tr['label'][0])
print('Decoded class from 1st label of DataFrame_Tr:',\
    ResponseEncoder.inverse_transform([DataFrame_Tr['label'][0]])[0])


#%% Instantiating datasets from dataframes

batch_size = 128
shuffle_buffer_size = np.max([batch_size,NumOfClasses])*2

load_from_cache = True

print('batch_size =',batch_size)
print(f'No. of training steps: {np.int(np.ceil(DataFrame_Tr.shape[0]/batch_size))}')
print(f'No. of Validation steps: {np.int(np.ceil(DataFrame_Val.shape[0]/batch_size))}')
print(f'No. of test steps: {np.int(np.ceil(DataFrame_Ts.shape[0]/batch_size))}')

if load_from_cache:
    Dataset_Tr = Dataset_from_Cache(cache_path=str(Dataset_dir.joinpath(f"{os.path.basename(DataFrame_Tr_path).split('.')[0]}_Cached")),\
        img_shape=original_img_shape, batch_size=batch_size,\
            shuffle=True, shuffle_buffer_size=shuffle_buffer_size, load_on_RAM=True)
    Dataset_Val = Dataset_from_Cache(cache_path=str(Dataset_dir.joinpath(f"{os.path.basename(DataFrame_Val_path).split('.')[0]}_Cached")),\
        img_shape=original_img_shape, batch_size=batch_size,\
            shuffle=False, shuffle_buffer_size=shuffle_buffer_size, load_on_RAM=True)
    # Dataset_Ts = Dataset_from_Cache(cache_path=str(MasterPath.joinpath('temp_datasets',DataFrame_Ts_path.parts[-1])),\
    #     img_shape=original_img_shape, batch_size=batch_size,\
    #         shuffle=False, shuffle_buffer_size=shuffle_buffer_size, load_on_RAM=True)
    print('Datasets created!')
else:
    cache_file = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    print('cache_time:',cache_file)
    Dataset_Tr = Dataset_from_Dataframe(dataframe=DataFrame_Tr,Img_dir=Img_dir,\
        batch_size=batch_size,shuffle=True,shuffle_buffer_size=shuffle_buffer_size,\
            cache_path=str(MasterPath.joinpath(f'TmpCache{cache_file}Tr')))        
    Dataset_Val = Dataset_from_Dataframe(dataframe=DataFrame_Val,Img_dir=Img_dir,\
        batch_size=batch_size,shuffle=False,shuffle_buffer_size=shuffle_buffer_size,\
            cache_path=str(MasterPath.joinpath(f'TmpCache{cache_file}Val')))
    # Dataset_Ts = Dataset_from_Dataframe(dataframe=DataFrame_Ts,Img_dir=Img_dir,\
    #     batch_size=batch_size,shuffle=False,shuffle_buffer_size=shuffle_buffer_size,\
    #         cache_path=str(MasterPath.joinpath(f'TmpCache{cache_file}Ts')))
    print('Datasets created!')


#%% Displaying sample images from each dataset

show_few_img = False
Num_of_sample = 10
if show_few_img:
    pos = 0
    for count, dataset in enumerate([Dataset_Tr,Dataset_Val,Dataset_Ts]):
        for (X,Y) in dataset.take(1):
            for num in range(Num_of_sample):
                pos += 1
                x=X[num,:,:,0]; y=Y[num]
                plt.subplot(3,Num_of_sample,pos).set_title(f"{['Train','Val','Test'][count]}: {y}")
                plt.imshow(x, cmap='gray', norm=matplotlib.colors.Normalize())


#%% Instantiating keras model

UseExistingArchitectureCores = True

models = {}
keras_preprocess_layers = {}
if UseExistingArchitectureCores:
    # models['mdl_Xception_232'] = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(232, 232, 3)) #input_shape=(71, 71, 3)
    # keras_preprocess_layers['mdl_Xception_232'] = tf.keras.applications.xception.preprocess_input
    # models['InceptionV3'] = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(75, 75, 3)) #input_shape=(75, 75, 3)
    # keras_preprocess_layers['InceptionV3'] = tf.keras.applications.inception_v3.preprocess_input
    models['mdl_InceptionResNetV2_172'] = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(172, 172, 3)) #input_shape=(75, 75, 3)
    keras_preprocess_layers['mdl_InceptionResNetV2_172'] = tf.keras.applications.inception_resnet_v2.preprocess_input
    # models['ResNet50V2'] = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(64, 64, 3)) #input_shape=(64, 64, 3)
    # keras_preprocess_layers['ResNet50V2'] = tf.keras.applications.resnet_v2.preprocess_input
    # models['DenseNet201'] = tf.keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(64, 64, 3)) #input_shape=(64, 64, 3)
    # keras_preprocess_layers['DenseNet201'] = tf.keras.applications.densenet.preprocess_input
    # models['NASNetLarge'] = tf.keras.applications.NASNetLarge(include_top=False, weights=None, input_shape=(64, 64, 3)) #input_shape=(64, 64, 3). NasNetLarge has to be trained from scratch since it doesn't support transfer learning unless the input_shape is (331, 331, 3).
    # keras_preprocess_layers['NASNetLarge'] = tf.keras.applications.nasnet.preprocess_input

    ModelKeys=list(models.keys())
    ModelsCreated = len(ModelKeys)
    print (str(ModelsCreated),' model/s loaded!')
else:
    print ('No model created. Load one from disk below!')    


#%% Saving model description

SaveModelDescription = False

if SaveModelDescription:
    for ModelKey in ModelKeys:
        model = models[ModelKey]
        Model_Path = os.path.join(MasterPath,str(ModelKey))        
        SaveModelDescript(model, save_dir=Model_Path, 
                          save_filename=str(ModelKey))        
    print ('Model descriptions saved!')


#%% Editing the core model

Edit_Core_Model = False
# SaveEditedModelDescription = True
# ModelKey = ModelKeys[0]

if Edit_Core_Model:
    model = models[ModelKey]
    New_Layers={'drop1':tf.keras.layers.Dropout(rate=0.1, name='drop1'),
                'drop2':tf.keras.layers.Dropout(rate=0.2, name='drop2'),
                'drop3':tf.keras.layers.Dropout(rate=0.3, name='drop3'),
                'drop4':tf.keras.layers.Dropout(rate=0.4, name='drop4'),
                'drop5':tf.keras.layers.Dropout(rate=0.5, name='drop5'),
               }

    IncomingLinks_2Axe=[-18, -12, -14, -8, -5, -1]   

    IncomingLinks_2Forge=[(New_Layers['drop1'], model.layers[-19]),
                          (model.layers[-18], New_Layers['drop1']),
                          (model.layers[-12], New_Layers['drop1']),
                          (New_Layers['drop2'], model.layers[-15]),
                          (model.layers[-14], New_Layers['drop2']),
                          (New_Layers['drop3'], model.layers[-9]),
                          (model.layers[-8], New_Layers['drop3']),
                          (New_Layers['drop4'], model.layers[-6]),
                          (model.layers[-5], New_Layers['drop4']),
                          (New_Layers['drop5'], model.layers[-2]),
                          (model.layers[-1], New_Layers['drop5']),
                         ]

    model_inputs=None
    model_outputs=None

    model = ModelEditor(model, New_Layers=New_Layers, IncomingLinks_2Axe=IncomingLinks_2Axe, 
                                IncomingLinks_2Forge=IncomingLinks_2Forge,
                                model_inputs=model_inputs, model_outputs=model_outputs)
    models[ModelKey] = model 
    
    # Save edited model description
    if SaveEditedModelDescription:
        Model_Path = os.path.join(MasterPath,str(ModelKey))        
        SaveModelDescript(model, save_dir=Model_Path, 
                          save_filename=str(ModelKey+'_edited'))        
        print ('Model descriptions saved!')


#%% Adding Top and Bottom layers to keras models instantiated above

AddTopAndBottomLayers = True
regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.001)
original_img_shape = original_img_shape #(w, h, No._of_Ch)
if AddTopAndBottomLayers:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        core = models[ModelKey]
        core.trainable = False
        
        In = tf.keras.Input(shape=original_img_shape, name="Preprocess_Input")
        adapter_layers = mdl_adapter_layers (Output_ImgShape=core.input_shape[1:])
        core_preprocess_layers = keras_preprocess_layers[ModelKey]
        Out = core_preprocess_layers(adapter_layers.Int_Rescaler(name='Ch_rescale_0_255_2')(\
            adapter_layers.Ch_Adjuster(kernel_regularizer=regularizer, name='Ch_adjuster')(adapter_layers.Int_Rescaler(name='Ch_rescale_0_255_1')(\
                adapter_layers.Im_Rotater(name='random_rotate')(\
                    adapter_layers.Im_Flipper(name='random_flip_HnV')(In))))))
        # Out = core_preprocess_layers(adapter_layers.Int_Rescaler(name='Ch_rescale_0_255_2')(\
        #     adapter_layers.Ch_Adjuster(name='Ch_adjuster')(adapter_layers.Int_Rescaler(name='Ch_rescale_0_255_1')(\
        #         adapter_layers.Im_Resizer(name='im_resize')(adapter_layers.Im_Rotater(name='random_rotate')(\
        #             adapter_layers.Im_Flipper(name='random_flip_HnV')(In)))))))
        mdl_preprocess = tf.keras.Model(inputs=In, outputs=Out, name='mdl_preprocess')
        
        In = tf.keras.Input(shape=core.output_shape[1:4], name="Features")
        GlbMaxPool = tf.keras.layers.GlobalMaxPool2D(name="GlbMaxPool")
        GlbAvgPool = tf.keras.layers.GlobalAveragePooling2D(name="GlbAvgPool")
        MaxAvgConcat = tf.keras.layers.Concatenate(axis=-1, name="MaxAvgConcat")
        Dropout1 = tf.keras.layers.Dropout(0.5, name="Feature_Dropout")
        Dense1 = tf.keras.layers.Dense(units=core.output_shape[-1], activation='relu', kernel_regularizer=regularizer, name="dense1")
        Dropout2 = tf.keras.layers.Dropout(0.5, name="Dropout2")
        predictions = tf.keras.layers.Dense(units=NumOfClasses, activation=None, kernel_regularizer=regularizer, name="predictions")
        Out=predictions(Dropout2(Dense1(Dropout1(MaxAvgConcat([GlbMaxPool(In),GlbAvgPool(In)])))))
        mdl_prediction = tf.keras.Model(inputs=In, outputs=Out, name='mdl_prediction')

        In = tf.keras.Input(shape=original_img_shape, name="Input_images")
        Out = mdl_prediction(core(mdl_preprocess(In)))
        models[ModelKey] = tf.keras.Model(inputs=In, outputs=Out, name=ModelKey)
    print ('Bottom and Top layers added!')    


#%% Saving model description

SaveModelDescription = True

if SaveModelDescription:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        model = models[ModelKey]
        Model_Path = os.path.join(MasterPath,str(ModelKey))        
        SaveModelDescript(model, save_dir=Model_Path, 
                          save_filename=str(ModelKey))        
    print ('Model descriptions saved!')


#%% Compiling the models

CompileModels = False

if CompileModels:
    ModelKeys = list(models.keys())
    for ModelKey in ModelKeys:
        model = models[ModelKey]
        model.compile(optimizer='adam', 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                        metrics=['accuracy'])
    print ('Models compiled!')


#%% Loading models from disk

LoadModelFromDisk = False

if LoadModelFromDisk:
    models['mdl_name'] = tf.keras.models.load_model('mdl_path')
    
    ModelsLoaded = len(models.keys()) - ModelsCreated
    print (str(ModelsLoaded),' models loaded from disk!')
else: print ('No model loaded from disk!')


#%% Adding Regularization to all regularizable layers

RegularizeTheModel = False
if RegularizeTheModel:
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.001)
    for ModelKey in ModelKeys:
        models[ModelKey]=RegularizeModel(models[ModelKey], regularizer, keep_weights=True)
else: print ('No alteration in regularization!')


#%% Summary of models

determine_initial_accuracies = False

if determine_initial_accuracies:
    ModelKeys=list(models.keys())
    print ('Total number of models = ',str(len(models.keys())))

    print ('Initial Train Loss and Accuracy')
    TrainEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(Dataset_Tr, verbose=0)
        TrainEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(TrainEval)) 

    print ('\nInitial Val Loss and Accuracy')
    ValEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(Dataset_Val, verbose=0)
        ValEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(ValEval)) 

    print ('\nInitial Test Loss and Accuracy')
    TestEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(Dataset_Ts, verbose=1) #verbose=0
        TestEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(TestEval)) 


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


#%% Training non-core layers

# Setting the core model as non-trainable
TrainNonCoreOnly = True
CoreModel_layer = -2

if TrainNonCoreOnly:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        model=models[ModelKey]
        model.layers[CoreModel_layer].trainable = False
        model.compile(optimizer='adam', 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                        metrics=['accuracy'])
        print (model.layers[CoreModel_layer].name,'(core model) has been set as non-trainable and', ModelKey, 'recompiled!')

if TrainNonCoreOnly:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        print (models[ModelKey].summary())


#%% Training non-core layers

if TrainNonCoreOnly:
    Epochs2TrainFor= 2000
    
    Start=time.perf_counter()
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        relevant_his_log=sorted([log for log in list(history.keys()) if (ModelKey+'_') in log])
        if len(relevant_his_log)>0:
            relevant_his_log = history[relevant_his_log[-1]]
            initial_epoch = relevant_his_log.epoch[-1] + 1
        else: initial_epoch = 0
        final_epoch = initial_epoch + Epochs2TrainFor

        ModelStart=time.perf_counter()
        print('\nTraining '+str(ModelKey)+' Non-Core...')
        Model_Path = os.path.join(MasterPath,str(ModelKey))
        model = models[ModelKey]
        sess_DateTime = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        history[ModelKey+'_'+sess_DateTime+'_NonCore']=train_model (
            model, Dataset_Tr, Dataset_Val, initial_epoch=initial_epoch, final_epoch=final_epoch,
            Model_Path=Model_Path, class_names=class_names)

        print('\n'+str(ModelKey)+' Non-Core trained! Training time = '+ str((time.perf_counter()-ModelStart)/60) + ' min!')
    print('\nTotal training time = '+ str((time.perf_counter()-Start)/(60*60)) + ' hr!')        
        

#%% Summary of Models

determine_accuracies = False

if determine_accuracies:
    print ('Total number of models = ',str(len(models.keys())))
    print ('Train Loss and Accuracy')
    TrainEval=[]
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(X_Train,Y_Train, verbose=0)
        TrainEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(TrainEval)) 

    print ('\nVal Loss and Accuracy')
    ValEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(X_Val,Y_Val, verbose=0)
        ValEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(ValEval)) 

    print ('\nTest Loss and Accuracy')
    TestEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(X_Test,Y_Test, verbose=0)
        TestEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(TestEval)) 


#%% Fine training additional layers

# Setting some layers of the core model as trainable
RegularizeTheModel = False
FineTrainCoreLayers = True
Change_DropoutRate = False
New_DropoutRate = 0.8

regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0.001)

# CoreModel_layer = -2
FineTuneOnwards = {
    'Xception':-5,#-7, #-16, #-7
    'InceptionV3':-5,#-31,
    'InceptionResNetV2':-5,#-19,
    'ResNet50V2':-5,#-13,
    'DenseNet201':-5,#-9, #-16
    'NASNetLarge':0,
}

if Change_DropoutRate:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        classi_model = models[ModelKey].layers[-1] 
        classi_model.layers[-2].rate = New_DropoutRate
        classi_model.layers[-4].rate = New_DropoutRate

if RegularizeTheModel:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        core_model = models[ModelKey].layers[CoreModel_layer] 
        core_model = RegularizeModel(models[ModelKey], regularizer, keep_weights=True)

if FineTrainCoreLayers:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        core_model = models[ModelKey].layers[CoreModel_layer]        
        core_model.trainable = True
        TrainableLayers_Key = [Key for Key in list(FineTuneOnwards.keys()) if Key in ModelKey][0]
        ref_mdl = eval(f'tf.keras.applications.{TrainableLayers_Key}(include_top=False, weights=None, input_shape={core_model.input_shape[1:]})')
        i_track=-1
        for layer in core_model.layers[:FineTuneOnwards[TrainableLayers_Key]]:
            i_track+=1
            layer.trainable =  False
        for layer in core_model.layers[FineTuneOnwards[TrainableLayers_Key]:]:
            i_track+=1
            layer.trainable = ref_mdl.layers[i_track].trainable
        del ref_mdl
        print ('Appropriate layers after',FineTuneOnwards[TrainableLayers_Key],'of',core_model.name,'have been set as trainable!')
        models[ModelKey].compile(optimizer=tf.keras.optimizers.Adam(1e-7), 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      metrics=['accuracy'])
        print (models[ModelKey].name,'has been recompiled!')

if FineTrainCoreLayers:
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        print (models[ModelKey].summary())


#%% Fine-tuning model

if FineTrainCoreLayers:
    Epochs2TrainFor= 2000

    Start=time.perf_counter()
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        relevant_his_log=sorted([log for log in list(history.keys()) if (ModelKey+'_') in log])
        if len(relevant_his_log)>0:
            relevant_his_log = history[relevant_his_log[-1]]
            initial_epoch = relevant_his_log.epoch[-1] + 1
        else: initial_epoch = 0
        final_epoch = initial_epoch + Epochs2TrainFor

        ModelStart=time.perf_counter()
        print('\nTraining '+str(ModelKey)+'...')
        Model_Path = os.path.join(MasterPath,str(ModelKey))
        model = models[ModelKey]
        sess_DateTime = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        history[ModelKey+'_'+sess_DateTime+'_FineTune']=train_model (
            model, Dataset_Tr, Dataset_Val, initial_epoch=initial_epoch, final_epoch=final_epoch,
            Model_Path=Model_Path, class_names=class_names)

        print('\n'+str(ModelKey)+' trained! Training time = '+ str((time.perf_counter()-ModelStart)/60) + ' min!')
    print('\nTotal training time = '+ str((time.perf_counter()-Start)/(60*60)) + ' hr!')  


#%% Summary of Models

determine_accuracies = False

if determine_accuracies:
    print ('Total number of models = ',str(len(models.keys())))
    print ('Train Loss and Accuracy')
    TrainEval=[]
    ModelKeys=list(models.keys())
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(X_Train,Y_Train, verbose=0)
        TrainEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(TrainEval)) 

    print ('\nVal Loss and Accuracy')
    ValEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(X_Val,Y_Val, verbose=0)
        ValEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(ValEval)) 

    print ('\nTest Loss and Accuracy')
    TestEval=[]
    for ModelKey in ModelKeys:
        Eval=models[ModelKey].evaluate(X_Test,Y_Test, verbose=0)
        TestEval.append(str(ModelKey)+' : '+str(Eval))
    print ('\n'.join(TestEval)) 


#%% Saving the latest version of each model

SaveLatestVersions = False
if SaveLatestVersions:
    sess = datetime.now().strftime("%Y%m%d-%H%M%S")
    for ModelKey in ModelKeys:
        print('\nSaving '+str(ModelKey))
        Save_Path = os.path.join(MasterPath,str(ModelKey),("LastModel_"+sess))
        # models[ModelKey].save(Save_Path)
        # tf.keras.models.save_model(models[ModelKey], Save_Path, \
        #     overwrite=False, include_optimizer=True)
        # Save_Path = os.path.join(MasterPath,str(ModelKey),("history_"+sess+'.pkl'))
        # with open(Save_Path,"wb") as history_file:
        #     pickle.dump(history,history_file)
    print('\nThe latest version of each model has been saved!')


#%%




