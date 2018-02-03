from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
import numpy as np
import re
from keras.utils.np_utils import to_categorical

def bn_word_batch_gen(coded_lines, batch_size, n_chars, max_len, eos_idx, bos_idx):
    while True:
        steps_per_epoch = len(coded_lines)/batch_size
        batch_counter = 0
        perm = np.random.permutation(len(coded_lines))
        for b in range(steps_per_epoch):
            start, end = b*batch_size, (b+1)*batch_size
            idxs = perm[start:end]
            batch = [coded_lines[j] for j in idxs]
            
            with_bos = [[bos_idx]+line_code for line_code in batch]
            X_batch = pad_sequences(with_bos, max_len+1, padding='post', truncating='post', value=eos_idx) # plus one for the bos
            padded_labels = pad_sequences(batch, max_len+1, padding='post', truncating='post', value=eos_idx)
            y_batch = np.asarray([to_categorical(a, num_classes=n_chars+1) for a in padded_labels])
            yield (X_batch, y_batch)
            batch_counter +=1
            
        assert batch_counter == len(coded_lines)/batch_size
        #print "There were {} batches of size {} in this epoch".format(batch_counter, batch_size)
        
def load_lm_model(model_name,  mode="best"):
    folder_name = "saved_lm_models"
    print "Loading model {}".format(model_name)
    
    print "Looking for files with mode \"{}\"".format(mode)

    filenames = os.listdir(folder_name)
    model_regex = r"^"+model_name+r"-([0-9]+)-([+-]?([0-9]*[.])?[0-9]+)\.h5"
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
        model=load_model(folder_name+'/'+latest)
        print "Done"
    elif mode == "best" and best:
        print "Best weights are from {}".format(best)
        print "Loading weights..."
        model=load_model(folder_name+'/'+best)
        print "Done"
    else:
        print "Didn't find any files for this model!"
        
        highest_epoch = 0
        
    return model, highest_epoch