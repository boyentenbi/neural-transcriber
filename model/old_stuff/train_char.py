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

# import pandas 

from process_data import *
import os

# Check what device is being used
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Necessary for the model?
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args # reorder the args because the order is shit
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


#print sys.argv[0] # prints python_script.py
# Script arguments
if len(sys.argv[1:]) == 2:
    model_name = sys.argv[1] 
    n_epochs = int(sys.argv[2]) 
else:
    assert len(sys.argv[1:])==1
    model_name = sys.argv[1]
    n_epochs = 10
    
name_regex = r"^densenet_char_b([0-9]+)\.l([0-9]+)\.f([0-9]+)\.k([0-9]+)\.([a-z]+)"
name_match = re.match(name_regex, model_name)

    
# Problem parameters
max_mfe_len = 1800
max_phrase_len = 300
n_energies = 40
n_letters = 27+1 # 26 chars, 1 space, 1 blank
blank_idx = n_letters - 1

# Model parameters
n_blocks = int(name_match.group(1))
layers_per_block = int(name_match.group(2))
n_filters = int(name_match.group(3))
kernel_size = int(name_match.group(4))
activation = name_match.group(5)

print "Regex for name matches with:", n_blocks, layers_per_block, n_filters, kernel_size, activation

max_dilation = 2**(layers_per_block-1)
block_rf = max_dilation * (kernel_size-1) + 1 + (kernel_size-1)/2 * (layers_per_block-1)
print "block_rf =", block_rf


if not os.path.isfile("saved_models/"+model_name):
    
    
    
    #model_name = "densenet_char_b{}.l{}.f{}.k{}.{}".format(n_blocks, layers_per_block, n_filters, kernel_size, activation)
    
    print "Creating model ", model_name
        
    # Input tensors include labels, input lengths and label lengths because we define the cost tensor explicitly
    log_mfes = Input(shape=(max_mfe_len, n_energies), name="log_mfes")
    labels = Input(name='the_labels', shape=[max_phrase_len], dtype='float32')
    labels_masked = (labels) # Should we have this?
    input_length = Input(name='input_length',shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Simple residual block without multiplicative gates or size-1 time convolutions
    def res_block(prev):
        r = prev
        # This gives an rf size of: 
        # max_dilation * (kernel_size-1) + 1 + (kernel_size-1)/2 * (layers_per_block-1)
        for x in range(layers_per_block):
            dilation_rate = 2**x
            linear = Conv1D(n_filters, 
                            kernel_size, 
                            padding='same', 
                            dilation_rate=dilation_rate, 
                            activation="linear")(r)
            batch_normed = BatchNormalization()(linear)
            a = Activation(activation)(batch_normed)
            r = concatenate([a, r])
        return r

    r = log_mfes
    for i in range(n_blocks):
        r = res_block(r)
    logits =  BatchNormalization()(Conv1D(n_letters, 
                                         kernel_size,
                                         padding='same', 
                                         dilation_rate=1, 
                                         activation="linear")(r))
    letter_probs = Activation("softmax", name = "letter_probs")(logits)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([letter_probs, labels_masked, input_length, label_length])

    # Create the model and compile
    model = Model(inputs=[log_mfes, labels, input_length, label_length], outputs=[loss_out, letter_probs])
    #model.summary()
    # clipnorm seems to speeds up convergence
    sgd = Adam(lr=0.001)
    
    print "Compiling..."
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss (y_pred is actually loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'letter_probs': lambda y_true, y_pred: 0*y_pred}, optimizer=sgd)

    

    folder_name = "saved_models"
    print "Writing model definition to", folder_name+"/"+model_name
    json= model.to_json()
    f=open(folder_name+"/"+model_name, "w")
    f.write(json)
    f.close()
    epochs_done=0
else:
    
    
    print "Loading model", model_name
    json_file = open("saved_models/"+model_name)
    json = json_file.read()
    model = keras.models.model_from_json(json)
    sgd = Adam(lr=0.001)
    print "Compiling..."
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss (y_pred is actually loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'letter_probs': lambda y_true, y_pred: 0*y_pred}, optimizer=sgd)

    print "Looking for weight files..."
    
    filenames = os.listdir("saved_weights")
    model_regex = r"^"+model_name+r"-([0-9]+)-([+-]?([0-9]*[.])?[0-9]+)\.hdf5"
    highest_epoch = None
    latest = None
    for k in filenames:
        match =  re.match(model_regex, k)
        if match:
            
            epochs_done = int(match.group(1))
            val_loss = float(match.group(2))
            print "File", k, "contains weights after {} epochs achieving validation loss: {}".format(epochs_done, val_loss)
            # If this is the first file, or if there is a more recent one
            if not highest_epoch or epochs_done > highest_epoch:
                highest_epoch = max(epochs_done, highest_epoch)
                latest = k
            else:
                continue
    if latest:
        print "Latest weights are from {}".format(latest)
    
        print "Loading weights..."
        model.load_weights('saved_weights/'+latest)
    else:
        print "Didn't find any weight files for this model!"
        epochs_done = 0
    
print "Preparing to train..."
print "Loading data from","Libri_data_lib" 
    
batch_size = 16
lib_fnames_train = ['Libri_data_lib/data_padded_{}.npz'.format(i) for i in range(4)]
lib_fnames_val = ['Libri_data_lib/data_padded_4.npz']
train_data_gen = batch_generator_chars(batch_size, lib_fnames_train, max_mfe_len, n_letters)
valid_data_gen = batch_generator_chars(batch_size, lib_fnames_val, max_mfe_len, n_letters)

total_data_size = 65040
val_split = 0.8
total_steps = int(np.ceil(float(total_data_size)/batch_size))
training_steps = int(np.ceil(total_data_size*0.8/batch_size ))
val_steps = int(np.ceil(total_data_size*0.2/batch_size ))
print "n training batches:", training_steps*batch_size, 
print "n validation batches:", val_steps*batch_size
print "n total batches:", total_steps*batch_size, 

weights_name = model_name+"-{epoch:02d}-{val_loss:.2f}.hdf5"
#checkpoint=keras.callbacks.ModelCheckpoint("saved_weights/"+weights_name, verbose=1, save_weights_only=True)

def batch_end_log(batch, logs):
    path = "training_logs/"+model_name+".npy"
    if os.path.isfile(path):
        existing_losses = np.load(path)
    else:
        existing_losses = np.asarray([])
#     print existing_losses
#     print logs['loss']
    new_losses = np.append(existing_losses, logs['loss'])
    np.save(path, new_losses)
    
def epoch_end_log(epoch, logs):
    filepath = "saved_weights/{}-{}-{}.hdf5".format(model_name, epoch+epochs_done, logs["val_loss"])
    model.save_weights(filepath)
    
log_callback = keras.callbacks.LambdaCallback(on_batch_end=batch_end_log)
checkpoint_callback = keras.callbacks.LambdaCallback(on_epoch_end=epoch_end_log)

print "batch-wise training log will be saved to:", "training_logs/"+model_name+".npy"

print "Starting training!"

# callback = keras.function([log_mfes], [])
history=model.fit_generator(train_data_gen, 
                            steps_per_epoch = training_steps, 
                            epochs=n_epochs, 
                            verbose = 1, 
                            callbacks=[checkpoint_callback, log_callback],
                            validation_data = valid_data_gen, 
                            validation_steps=val_steps)
