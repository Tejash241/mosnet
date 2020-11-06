"""
Original Author: Alex Cannan
Modifying Author: You!
Date Imported: 
Purpose: This file contains a script meant to train a model.
"""

import os
import sys

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
n_epochs = 10
batch_size = 32

# TODO: Read training data and split into training and validation sets
mos_list = utils.read_list(os.path.join(DATA_DIR, 'mos_list.txt'))
mos_train, mos_valid = random.

# TODO: Initialize and compile model
MOSNet = model.CNN()
model = MOSNet.build()

# TODO: Start fitting model using utils.data_generator
