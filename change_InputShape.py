

#%%

import tensorflow as tf

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
