import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from config import *


def build_model(freeze_base=True):
    inp  = Input(shape=(224, 224, 3))
    base = ResNet50V2(weights='imagenet', include_top=False, input_tensor=inp)

    base.trainable = not freeze_base

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = Dropout(0.6)(x)
    x = Dense(N_CLASSES, activation='linear', dtype='float32')(x)
    x = tf.keras.layers.Softmax(dtype='float32')(x)

    model = Model(inputs=inp, outputs=x)
    return model, base

def unfreeze_base_layers(base, n_layers=5):
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
        print(f"Layer {layer.name} unfrozen")


