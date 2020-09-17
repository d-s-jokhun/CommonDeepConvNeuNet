

#%%

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import zipfile
# import Lambda_functions
import multiprocessing as mp


#%%

# class Dataset_from_Dataframe(tf.data.Dataset):
    
    # def img_fetcher (Img_dir, rel_path):
    #     rel_path=rel_path[2:-1]
    #     with zipfile.ZipFile(os.path.join(Img_dir.decode(),os.path.dirname(rel_path.decode()))) as archive:
    #         img = np.array(Image.open(io.BytesIO(archive.read(os.path.basename(rel_path).decode())))).astype(np.float32)
    #     return np.expand_dims(np.array(img),-1)

    # def fetch_img ():

    # def __new__(cls, dataframe, Img_dir, batch_size, shuffle=False, shuffle_buffer_size=None):
    #     AUTOTUNE = tf.data.experimental.AUTOTUNE
    #     if type(shuffle_buffer_size)==type(None):
    #         shuffle_buffer_size=dataframe.shape[0]

    #     def fetch (rel_path,label):
    #         img = tf.numpy_function(cls.img_fetcher, [Img_dir, rel_path], tf.float32)
    #         return (img,label)

    #     if shuffle:
    #         return tf.data.Dataset.from_tensor_slices(\
    #             (dataframe['rel_path'], \
    #     dataframe['label']))\
    #         .map(fetch)\
    #             .cache()\
    #                 .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)\
    #                     .batch(batch_size)\
    #                         .prefetch(1)

    #     elif not shuffle:
    #         return tf.data.Dataset.from_tensor_slices(\
    #             (dataframe['rel_path'], \
    #     dataframe['label']))\
    #         .map(fetch)\
    #             .batch(batch_size)\
    #                 .cache()\
    #                     .prefetch(1) 


#%%

class Dataset_from_Dataframe(tf.data.Dataset):
    
    def img_fetcher (Img_dir, rel_path):
        with zipfile.ZipFile(os.path.join(Img_dir.decode(),os.path.dirname(rel_path.decode()))) as archive:
            img = np.array(Image.open(io.BytesIO(archive.read(os.path.basename(rel_path).decode())))).astype(np.float32)
        return img

    def fetch_img(rel_path_batch, Img_dir, img_fetcher=img_fetcher):
        rel_path_batch = [rel_path[2:-1] for rel_path in rel_path_batch]
        with mp.Pool() as pool:
            img_batch = pool.starmap(img_fetcher,\
                zip([Img_dir for rel_path in rel_path_batch],\
                    rel_path_batch))
        return np.expand_dims(np.array(img_batch),-1)

    def __new__(cls, dataframe, Img_dir, batch_size, shuffle=False, shuffle_buffer_size=None,cache_path=None):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if type(shuffle_buffer_size)==type(None):
            shuffle_buffer_size=dataframe.shape[0]

        def fetch (rel_path_batch,label_batch):
            img_batch = tf.numpy_function(cls.fetch_img, [rel_path_batch, Img_dir], tf.float32)
            return (img_batch,label_batch)

        def fetch_as_DS (rel_path_batch,label_batch):
            return tf.data.Dataset.from_tensor_slices((fetch(rel_path_batch,label_batch)))
        
        if shuffle:
            return tf.data.Dataset.from_tensor_slices(\
                (dataframe['rel_path'], \
        dataframe['label']))\
            .batch(dataframe.shape[0])\
                .flat_map(fetch_as_DS)\
                    .cache(cache_path)\
                        .shuffle(buffer_size=shuffle_buffer_size,reshuffle_each_iteration=True)\
                            .batch(batch_size)\
                                .prefetch(1)
        elif not shuffle:
            return tf.data.Dataset.from_tensor_slices(\
                (dataframe['rel_path'], \
        dataframe['label']))\
            .batch(batch_size)\
                .map(fetch,num_parallel_calls=AUTOTUNE)\
                    .cache(cache_path)\
                        .prefetch(1)
                                    

#%%



