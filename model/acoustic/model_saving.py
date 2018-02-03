#print "Importing modules..."
import sys
import re
# import tensorflow as tf
# sess = tf.Session()
import keras
from keras import backend as K
# K.set_session(sess)
import h5py

from keras.models import Model
from keras.optimizers import SGD, Adam

import numpy as np

# import pandas 

import os

def load_model(model_name, mode="best"):
    print "Loading model {}".format(model_name)
    json_file = open("saved_models/"+model_name)
    json = json_file.read()
    model = keras.models.model_from_json(json)
    sgd = Adam(lr=0.001)
    print "Compiling..."
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss (y_pred is actually loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'phoneme_probs': lambda y_true, y_pred: 0*y_pred}, optimizer=sgd)

    print "Looking for weight files with mode \"{}\"".format(mode)

    filenames = os.listdir("saved_weights")
    model_regex = r"^"+model_name+r"-([0-9]+)-([+-]?([0-9]*[.])?[0-9]+)\.hdf5"
    highest_epoch = None
    best_val_loss = None
    latest = None
    best = None
    for k in filenames:
        match =  re.match(model_regex, k)
        if match:

            epochs_done = int(match.group(1))
            val_loss = float(match.group(2))
            print "File", k, "contains weights after {} epochs achieving validation loss: {}".format(epochs_done, val_loss)
            # If this is the first file, or if there is a more recent one
            if not latest or epochs_done > highest_epoch:
                highest_epoch = epochs_done
                latest = k
                
            if not best or val_loss < best_val_loss:
                best_val_loss = val_loss
                best = k
                
            
    if mode=="latest" and latest:
        print "Latest weights are from {}".format(latest)
        print "Loading weights..."
        model.load_weights('saved_weights/'+latest)
        print "Done"
    elif mode == "best" and best:
        print "Best weights are from {}".format(best)
        print "Loading weights..."
        model.load_weights('saved_weights/'+best)
        print "Done"
    else:
        print "Didn't find any weight files for this model!"
        
        highest_epoch = 0
        
    return model, highest_epoch

def load_model_char(model_name, mode="best"):
    print "Loading model {}".format(model_name)
    json_file = open("saved_models/"+model_name)
    json = json_file.read()
    model = keras.models.model_from_json(json)
    sgd = Adam(lr=0.001)
    print "Compiling..."
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss (y_pred is actually loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'char_probs': lambda y_true, y_pred: 0*y_pred}, optimizer=sgd)

    print "Looking for weight files with mode \"{}\"".format(mode)

    filenames = os.listdir("saved_weights")
    model_regex = r"^"+model_name+r"-([0-9]+)-([+-]?([0-9]*[.])?[0-9]+)\.hdf5"
    highest_epoch = None
    best_val_loss = None
    latest = None
    best = None
    for k in filenames:
        match =  re.match(model_regex, k)
        if match:

            epochs_done = int(match.group(1))
            val_loss = float(match.group(2))
            print "File", k, "contains weights after {} epochs achieving validation loss: {}".format(epochs_done, val_loss)
            # If this is the first file, or if there is a more recent one
            if not latest or epochs_done > highest_epoch:
                highest_epoch = epochs_done
                latest = k
                
            if not best or val_loss < best_val_loss:
                best_val_loss = val_loss
                best = k
                
            
    if mode=="latest" and latest:
        print "Latest weights are from {}".format(latest)
        print "Loading weights..."
        model.load_weights('saved_weights/'+latest)
        print "Done"
    elif mode == "best" and best:
        print "Best weights are from {}".format(best)
        print "Loading weights..."
        model.load_weights('saved_weights/'+best)
        print "Done"
    else:
        print "Didn't find any weight files for this model!"
        
        highest_epoch = 0
        
    return model, highest_epoch