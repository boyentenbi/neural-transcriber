print "Importing modules..."
import sys
import re
# import tensorflow as tf
# sess = tf.Session()
import keras
from keras import backend as K
# K.set_session(sess)
import h5py

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Embedding, GRU, Masking, Dropout, Lambda, Flatten, concatenate, Conv1D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

import numpy as np
import csv
# import pandas 

#from process_data import *
#from model_saving import load_model_char
import os

from lm_train_helpers import bn_word_batch_gen, load_lm_model

# Check what device is being used
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#print sys.argv[0] # prints python_script.py
# Script arguments
if len(sys.argv[1:]) == 2:
    model_name = sys.argv[1] 
    n_epochs = int(sys.argv[2]) 
else:
    assert len(sys.argv[1:])==1
    model_name = sys.argv[1]
    n_epochs = 10
    
name_regex = r"^lm_char_e([0-9]+)\.b([0-9]+)\.l([0-9]+)\.f([0-9]+)\.k([0-9]+)\.([a-z]+)"
name_match = re.match(name_regex, model_name)

assert name_match

# Problem parameters
max_len = 280
n_chars = 27 # 27 chars including space, 1 eos
eos_idx = n_chars 
bos_idx = n_chars + 1

# Model parameters
embed_size = int(name_match.group(1))
n_blocks = int(name_match.group(2))
layers_per_block = int(name_match.group(3))
n_filters = int(name_match.group(4))
kernel_size = int(name_match.group(5))
activation = name_match.group(6)

assert kernel_size%2==1 # odd kernel size helps compute block rf easily... but otherwise no benefit
print "Regex for name matches with:", n_blocks, layers_per_block, n_filters, kernel_size, activation

block_rf = 1
for i in range(layers_per_block):
    dilation_rate = 2**i
    block_rf += dilation_rate*(kernel_size-1)
print "block_rf =", block_rf

if not os.path.isfile("saved_lm_models/"+model_name):
    
    # Simple residual block without multiplicative gates or size-1 time convolutions
    def causal_res_block(prev):
        r = prev
        for x in range(layers_per_block):
             
            dilation_rate = 2**x
            linear = Conv1D(n_filters, 
                            kernel_size, 
                            padding='causal', 
                            dilation_rate=dilation_rate, 
                            activation="linear")(r)
            batch_normed = BatchNormalization()(linear)
            a = Activation(activation)(batch_normed)
            r = concatenate([a, r])
        return r

    # print "Creating model ", model_name

    # Input tensors include labels, input lengths and label lengths because we define the cost tensor explicitly
    chars = Input(shape=(max_len+1,), name="chars")
    embed = Embedding(input_dim = n_chars + 2, output_dim = embed_size, name = "embed")(chars)
    # print keras.backend.shape(embed)

    r = embed
    for i in range(n_blocks):
        r = causal_res_block(r)
    logits = BatchNormalization()(Conv1D(n_chars+1, 
                                         kernel_size,
                                         padding='causal', 
                                         dilation_rate=1, 
                                         activation="linear")(r))
    char_probs = Activation("softmax", name = "char_probs")(logits)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer

    # Create the model and compile
    model = Model(inputs=[chars], outputs=[char_probs])
    filepath = "saved_lm_models/{}-{}-{}.h5".format(model_name, 0, 999999999.)
    model.save(filepath)
    highest_epoch = 0
else:
    model, highest_epoch = load_lm_model(model_name, mode="latest")

model.summary()
sgd = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print "Preparing to train..."
    
#lib_fnames_train = ['Libri_data_lib/data_padded_phon_{}.npz'.format(i) for i in range(8)]
#lib_fnames_val = ['Libri_data_lib/data_padded_phon_{}.npz'.format(i) for i in [8,9]]

print "Loading data... (this will take a few minutes)"
train_coded_lines = np.load('coded_clean_texts.npy')
valid_coded_lines = np.load('coded_clean_texts_heldout.npy')
print "Finished loading data. There are {} training examples and {} validation examples".format(len(train_coded_lines), len(valid_coded_lines))

batch_size = 256
print "Using a batch size of {}".format(batch_size)
train_steps_per_epoch = len(train_coded_lines)/batch_size
valid_steps_per_epoch = len(valid_coded_lines)/batch_size
train_data_gen = bn_word_batch_gen(train_coded_lines, batch_size, n_chars, max_len, eos_idx, bos_idx)
valid_data_gen = bn_word_batch_gen(valid_coded_lines, batch_size, n_chars, max_len, eos_idx, bos_idx)


def epoch_end_log(epoch, logs):
    filepath = "saved_lm_models/{}-{}-{}.h5".format(model_name, epoch+highest_epoch, logs["val_loss"])
    model.save(filepath)
    
checkpoint_callback = keras.callbacks.LambdaCallback(on_epoch_end=epoch_end_log)

print "Starting training!"

# callback = keras.function([log_mfes], [])
history=model.fit_generator(train_data_gen, 
                            steps_per_epoch = train_steps_per_epoch, 
                            epochs=n_epochs, 
                            verbose = 1, 
                            callbacks=[checkpoint_callback],
                            validation_data = valid_data_gen, 
                            validation_steps=valid_steps_per_epoch)
