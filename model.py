"""
Original Author: Alex Cannan
Modifying Author: You!
Date Imported: 
Purpose: This file contains the CNN MOSNet model described in [1].

[1]: https://arxiv.org/abs/1904.08352
"""

from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D, TimeDistributed


FREQ_DIM = 1025  # 1+n_fft/2 -> 1+2048/2 = 1025


class CNN(object):

    def __init__(self, freq_dim=FREQ_DIM):
        self.freq_dim = freq_dim
        print('CNN init')

    def build(self):
        _input = keras.Input(shape=(None, self.freq_dim))

        re_input = layers.Reshape((-1, self.freq_dim, 1), 
                                  input_shape=(-1, self.freq_dim))(_input)

        # CNN
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 3),
                        activation='relu', padding='same'))(conv1)

        conv2 = (Conv2D(32, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 3),
                        activation='relu', padding='same'))(conv2)

        conv3 = (Conv2D(64, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 3),
                        activation='relu', padding='same'))(conv3)

        conv4 = (Conv2D(128, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 1),
                        activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 3),
                        activation='relu', padding='same'))(conv4)

        # DNN
        flatten = TimeDistributed(layers.Flatten())(conv4)
        dense1 = TimeDistributed(Dense(64, activation='relu'))(flatten)
        dense1 = Dropout(0.3)(dense1)

        frame_score = TimeDistributed(Dense(1), name='frame')(dense1)

        average_score = layers.GlobalAveragePooling1D(name='avg')(frame_score)

        model = Model(outputs=[average_score, frame_score], inputs=_input)

        return model
