
#%%
import tensorflow as tf
import numpy as np


#%%

class Dataset_from_Cache(tf.data.Dataset):
    
    def __new__(cls, cache_path, img_shape, batch_size, shuffle=False, shuffle_buffer_size=None, load_on_RAM=False):
        
        dataset = tf.data.Dataset.from_tensor_slices(\
            (np.zeros((1,*img_shape)).astype(np.float32),np.zeros((1,)).astype(np.int64)))\
                .cache(cache_path)
        def gen (dataset=dataset):
            for x,y in dataset:
                yield (x,y)

        if shuffle:
            if load_on_RAM:
                return tf.data.Dataset.from_generator(gen,(tf.float32, tf.int64),\
                    (tf.TensorShape(img_shape), tf.TensorShape([])))\
                        .cache()\
                            .shuffle(buffer_size=shuffle_buffer_size,reshuffle_each_iteration=True)\
                                .batch(batch_size)\
                                    .prefetch(1)
            elif not load_on_RAM:
                return tf.data.Dataset.from_generator(gen,(tf.float32, tf.int64),\
                    (tf.TensorShape(img_shape), tf.TensorShape([])))\
                        .shuffle(buffer_size=shuffle_buffer_size,reshuffle_each_iteration=True)\
                            .batch(batch_size)\
                                .prefetch(1)

        elif not shuffle:
            if load_on_RAM:
                return tf.data.Dataset.from_generator(gen,(tf.float32, tf.int64),\
                    (tf.TensorShape(img_shape), tf.TensorShape([])))\
                        .cache()\
                            .batch(batch_size)\
                                .prefetch(1)
            elif not load_on_RAM:
                return tf.data.Dataset.from_generator(gen,(tf.float32, tf.int64),\
                    (tf.TensorShape(img_shape), tf.TensorShape([])))\
                        .batch(batch_size)\
                            .prefetch(1)
            
            
#%%

