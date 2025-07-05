import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def residual_block(x, filters):
    shortcut = TimeDistributed(Conv2D(filters, (1, 1), padding='same'))(x)
    x = TimeDistributed(
        Conv2D(filters, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(SpatialDropout2D(0.3))(x)
    x = TimeDistributed(Conv2D(filters, (3, 3), padding='same'))(x)
    x = Add()([x, shortcut])
    return Activation('relu')(x)


def build_vionet(input_shape=(20, 64, 64, 3)):
    inputs = Input(shape=input_shape)
    x = residual_block(inputs, 32)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = residual_block(x, 64)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = ConvLSTM2D(48, (3, 3), padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(96, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)
