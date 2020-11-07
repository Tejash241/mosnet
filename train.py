"""
Original Author: Alex Cannan
Modifying Author: You!
Date Imported: 
Purpose: This file contains a script meant to train a model.
"""

import os
import sys
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import model
import utils

# dir setup
DATA_DIR = os.path.join('.', 'data')
BIN_DIR = os.path.join(DATA_DIR, 'bin')
OUTPUT_DIR = os.path.join('.', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TODO: Set hyperparameters
random.seed(42)
np.random.seed(42)
N_EPOCHS = 10
BATCH_SIZE = 2
PERC_TRAIN = 0.75


# TODO: Read training data and split into training and validation sets
class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_paths, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_paths)

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        X, (y_mos, y_frame) = utils.data_generator(self.file_paths, BIN_DIR, index, batch_size=self.batch_size)
        return X, {'avg': y_mos, 'frame': y_frame}


mc = keras.callbacks.ModelCheckpoint('output/weights{epoch+1:02d}.h5', period=1)

mos_list = utils.read_list(os.path.join(DATA_DIR, 'mos_list.txt'))
print(f'{len(mos_list)} lines read from mos_list.txt')
train_idx = np.random.randint(0, len(mos_list), int(PERC_TRAIN*len(mos_list)))
mos_list = np.array(mos_list)
mos_train = DataGenerator(mos_list[train_idx], BATCH_SIZE, True)
mos_valid = DataGenerator(np.delete(mos_list, train_idx), BATCH_SIZE, False)
print(f'{len(mos_train)} training and {len(mos_valid)} validation batches separated from the dataset')

# TODO: Initialize and compile model
MOSNet = model.CNN()
model = MOSNet.build()
model.load_weights('output/weights01.h5')
print('Model loaded')

# TODO: Start fitting model using utils.data_generator
def loss_avg(y_avg, pred_avg):
    loss_per_utterance = tf.pow(y_avg - pred_avg, 2)
    return tf.reduce_mean(loss_per_utterance, axis=0)


def loss_frame(y_frame, pred_frame, alpha=1):
    # alpha is defaulted to 1 acc to the paper in section 3.2
    loss_per_frame = alpha * tf.reduce_mean(tf.pow(y_frame - pred_frame, 2), axis=1)
    return tf.reduce_mean(loss_per_frame, axis=0)


model.compile(optimizer='adam', loss={'avg': loss_avg, 'frame': loss_frame})
history = model.fit(x=mos_train, validation_data=mos_valid, callbacks=[mc], epochs=N_EPOCHS)
print(history)