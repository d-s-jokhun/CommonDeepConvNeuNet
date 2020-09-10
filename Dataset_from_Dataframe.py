

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

    def __new__(cls, dataframe, Img_dir, batch_size, shuffle=False, shuffle_buffer_size=None):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if type(shuffle_buffer_size)==type(None):
            shuffle_buffer_size=dataframe.shape[0]

        def fetch (rel_path_batch,label_batch):
            img_batch = tf.numpy_function(cls.fetch_img, [rel_path_batch, Img_dir], tf.float32)
            return (img_batch,label_batch)

        if shuffle:
            return tf.data.Dataset.from_tensor_slices(\
                (dataframe['rel_path'], \
        dataframe['label']))\
            .cache()\
                .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)\
                    .batch(batch_size)\
                        .map(fetch, num_parallel_calls=AUTOTUNE)\
                            .prefetch(AUTOTUNE)
        elif not shuffle:
            return tf.data.Dataset.from_tensor_slices(\
                (dataframe['rel_path'], \
        dataframe['label']))\
            .batch(batch_size)\
                .cache()\
                    .map(fetch, num_parallel_calls=AUTOTUNE)\
                        .prefetch(AUTOTUNE)
                                    

# %%

# class Dataset_from_Dataframe(tf.data.Dataset):
    
#     def img_fetcher (Img_dir, rel_path):
#         with zipfile.ZipFile(os.path.join(Img_dir.decode(),os.path.dirname(rel_path.decode()))) as archive:
#             img = np.array(Image.open(io.BytesIO(archive.read(os.path.basename(rel_path).decode())))).astype(np.float32)
#         return img

#     def fetch_img(rel_path_batch, Img_dir, img_fetcher=img_fetcher):
#         rel_path_batch = [rel_path[2:-1] for rel_path in rel_path_batch]
#         with mp.Pool() as pool:
#             img_batch = pool.starmap(img_fetcher,\
#                 zip([Img_dir for rel_path in rel_path_batch],\
#                     rel_path_batch))
#         return np.expand_dims(np.array(img_batch),-1)

#     def resize_img (img_batch, img_size):
#         return tf.image.resize_with_pad(img_batch, target_height=img_size[1], target_width=img_size[0],\
#             method=tf.image.ResizeMethod.BILINEAR, antialias=True)

#     def augment_img (img_batch):
#         return tf.keras.Sequential([
#             tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#             tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, fill_mode='constant', interpolation='bilinear')])\
#                 (img_batch, training=True)

#     def Ch_Norm_255(img_batch,label_batch):
#         img_batch=Lambda_functions.Ch_Norm_255(img_batch)
#         return (img_batch, label_batch)

#     def __new__(cls, dataframe, Img_dir, batch_size, shuffle=False, shuffle_buffer_size=None, img_size=None, augmentation=False):
#         AUTOTUNE = tf.data.experimental.AUTOTUNE
#         if type(shuffle_buffer_size)==type(None):
#             shuffle_buffer_size=dataframe.shape[0]

#         def fetch (rel_path_batch,label_batch):
#             img_batch = tf.numpy_function(cls.fetch_img, [rel_path_batch, Img_dir], tf.float32)
#             return (img_batch,label_batch)

#         def resize (img_batch,label_batch):
#             if type(img_size)!=type(None):
#                 img_batch = tf.numpy_function(cls.resize_img, [img_batch,img_size], tf.float32)
#             return (img_batch,label_batch)

#         def augment (img_batch,label_batch):
#             if augmentation == True:
#                 img_batch = tf.numpy_function(cls.augment_img, [img_batch], tf.float32)
#             return (img_batch,label_batch)

#         if shuffle:
#             return tf.data.Dataset.from_tensor_slices(\
#                 (dataframe['rel_path'], \
#         dataframe['label']))\
#             .cache()\
#                 .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)\
#                     .batch(batch_size)\
#                         .map(fetch)\
#                             .map(resize)\
#                                 .map(augment, num_parallel_calls=AUTOTUNE)\
#                                     .map(cls.Ch_Norm_255, num_parallel_calls=AUTOTUNE)\
#                                         .prefetch(AUTOTUNE)
#         elif not shuffle:
#             return tf.data.Dataset.from_tensor_slices(\
#                 (dataframe['rel_path'], \
#         dataframe['label']))\
#             .cache()\
#                 .batch(batch_size)\
#                     .map(fetch)\
#                         .map(resize)\
#                             .map(augment, num_parallel_calls=AUTOTUNE)\
#                                 .map(cls.Ch_Norm_255, num_parallel_calls=AUTOTUNE)\
#                                     .prefetch(AUTOTUNE)

