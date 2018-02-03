import re
import os
import numpy as np
import nltk
import itertools
from python_speech_features import fbank
import soundfile as sf
import multiprocessing as mp
import signal
import string

import csv
import keras

def get_names(folder_name, reg_expr):
    file_names = os.listdir(folder_name)
    data_names = []
    for name in file_names:
        match = re.search(reg_expr, name)
        if match:
            data_names.append(match.group(1))
    return data_names

def import_flac_dat(data_names, folder_name):
    return import_audio_dat(data_names, folder_name, ".flac")

#imports a .wav file as a tuple (sample_rate, signal)
def import_wav_dat(data_names, folder_name):
    return import_audio_dat(data_names, folder_name, ".wav")

def import_audio_dat(data_names, folder_name, extension):
    audio_dats = []
    for name in data_names:
        file_path = os.path.join(folder_name, name + extension)
        audio_dats.append(audio_read(file_path))
    return audio_dats


def audio_read(dat_file_path):
    (signal, rate) = sf.read(dat_file_path)
    return (rate, signal)

#gets the mel frequency energy forms of a folder of audio files
def get_MFEs(folder_name, reg_expr, extension):
    audio_dats = import_audio_dat(get_names(folder_name, reg_expr), folder_name)
    return [mel_transform(signal, sample_rate = rate) for (rate, signal) in audio_dats]

#does a mel transform on a signal
def mel_transform(signal, sample_rate = 8000, pre_emphasis = 0.97
                 , frame_size = 0.025, frame_stride = 0.01, window_func = np.hamming
                 , N_FFT = 512, nfilt = 40, mean_normalised = True):
    feat, energy = fbank(signal, samplerate = sample_rate, winlen = frame_size
                        , winstep = frame_stride, nfilt = nfilt, nfft = N_FFT
                        , preemph = pre_emphasis, winfunc = np.hamming)
    return np.log(feat)

    #apply pre-emphasis filter
    # emph_sig = np.append(signal[0], signal[1:] - pre_emphasis*signal[:-1])
    # frame_length = frame_size*sample_rate #converting from seconds to samples
    # frame_step = frame_stride*sample_rate
    # signal_length = len(emph_sig)
    # frame_length = int(round(frame_length))
    # frame_step = int(round(frame_step))
    # num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    # pad_signal_length = num_frames * frame_step + frame_length
    # z = np.zeros((pad_signal_length - signal_length))
    #
    # #makes all frames have equal number of samples
    # pad_signal = np.append(emph_sig, z)
    #
    # #slices singal into frames
    #
    # indices = (np.tile(np.arange(0, frame_length), (num_frames, 1)) +
    #           np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T)
    # frames = pad_signal[indices.astype(np.int32, copy=False)]
    #
    # #applies window function. Default is hamming window
    # frames *= window_func(frame_length)
    # fft_frames = np.fft.rfft(frames, N_FFT)
    # pow_frames = ((1.0 / N_FFT) * (np.absolute(fft_frames) ** 2))
    # filter_banks = apply_triag_filters(pow_frames, sample_rate, N_FFT, nfilt)
    # if mean_normalised:
    #     filter_banks -= (np.mean(filter_banks, axis = 0) + 1e-8)
    #
    # return filter_banks


'''
#applies triangular filters to fourier transformed frames
def apply_triag_filters(pow_frames, sample_rate, N_FFT = 512, nfilt = 40):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    f_image = np.floor((N_FFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(N_FFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(f_image[m - 1])   # left
        f_m = int(f_image[m])             # center
        f_m_plus = int(f_image[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_image[m - 1]) / (f_image[m] - f_image[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_image[m + 1] - k) / (f_image[m + 1] - f_image[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks) # dB
    return filter_banks
'''

'''
#stacks frames into stacks of size stack_size
def stack_frames(mfe, stack_size = 8):
    frame_stack_list = []
    for ind in range(len(mfe[:, 0]) - stack_size):
        frame_stack = mfe[ind:ind+stack_size, :]

        #converts each framestack into a vector
        frame_stack_list.append(frame_stack.flatten())
    return frame_stack_list

#converts a batch of MFEs into a batch of framestacks lists
def convert_to_stack(mfe_batch, stack_size = 8, stack_stride = 3):
    #stores the batch of framestacks
    frame_array_batch = []

    #for each mfe, converts it into an array representing framestacks
    for mfe in mfe_batch:
        frame_stack_list = stack_frames(mfe, stack_size)

         #takes every stack_stride in the list
        frame_stack_list = frame_stack_list[:: stack_stride]
        frame_stack_ar = np.array(frame_stack_list)
        frame_array_batch.append(frame_stack_ar)

    return frame_array_batch
'''
#given a sentence in english and a phoneme dictionary, outputs a list of phonemes
def obtain_phonemes(sentence, dictionary):
    sentence_phonemes = []
    for word in sentence:
        word = word.lower()
        if word in dictionary:
            #nltk.corpus.cmudict.dict() gives a list of lists, giving both
            #the american and english pronounciations
            phonemes = dictionary[word][0]
            stripped_phonemes = map(strip_stresses, phonemes)
            sentence_phonemes.extend(stripped_phonemes)
        else:
            #print "{} is not in the dictionary, can't get phonemes!".format(word)
            return None
    return sentence_phonemes

#removes stresses from vowels
def strip_stresses(phoneme):
    main_phon = phoneme[:-1]
    last_letter = phoneme[-1]
    if last_letter.isnumeric():
        return main_phon
    else:
        return phoneme

#returns a dictionary of tuples (mfe, sentence_phonemes)
def get_labelled_dat_dict(label_suffix, folder_name = "./", prefix = "", extension = ".wav"):
    reg_extension = "\\" + extension
    label_reg_expr = r"^(" + prefix + r".*)" + reg_extension + r"\." + label_suffix + r"$"
    label_names = get_names(folder_name, label_reg_expr)
    return convert_to_dat_dict(label_suffix, folder_name, label_names, extension)

#given a list of names, converts to a dictionary of tuples
def convert_to_dat_dict(label_suffix, folder_name, label_names, extension):
    labelled_dat_dict = {}
    for name in label_names:
        label_file_path = os.path.join(folder_name, name + extension + "." + label_suffix)
        f = open(label_file_path, "r")
        sentence = f.read()
        sentence_phonemes = obtain_phonemes(sentence.split(), nltk.corpus.cmudict.dict())
        if sentence_phonemes == None:
            pass
        else:
            dat_file_path = os.path.join(folder_name, name + extension)
            (rate, signal) = audio_read(dat_file_path)
            mfe = mel_transform(signal, sample_rate = rate)
            labelled_dat_dict[name] = (mfe, sentence_phonemes)
    return labelled_dat_dict

def code_label(s, D):
    s_clean = clean_sentence(s)
    assert s_clean[-1]=="\n"
    words = s_clean[:-1].split()
    phonemes = obtain_phonemes(words, D)
    if phonemes:
        return code_phonemes(phonemes)
    else:
        return None

def clean_label(s):
    # clean white space
    s_ = ' '.join(s.split())

    # remove punctuation and make lower case
    s_ = s_.translate(None, string.punctuation).lower()

    return s_

#returns two lists, (mfes, labels)
def get_labelled_dat(label_suffix, folder_name = "./", prefix = "", extension = ".wav"):
    labelled_dat_dict = get_labelled_dat_dict(label_suffix, folder_name, prefix, extension)
    return zip(*(labelled_dat_dict.values()))

#converts a list of phonemes to codes
def code_phonemes(phoneme_list):
    phoneme_dict =   {#' ': 0,
                     u'AA': 0,
                     u'AE': 1,
                     u'AH': 2,
                     u'AO': 3,
                     u'AW': 4,
                     u'AY': 5,
                     u'B': 6,
                     u'CH': 7,
                     u'D': 8,
                     u'DH': 9,
                     u'EH': 10,
                     u'ER': 11,
                     u'EY': 12,
                     u'F': 13,
                     u'G': 14,
                     u'HH': 15,
                     u'IH': 16,
                     u'IY': 17,
                     u'JH': 18,
                     u'K': 19,
                     u'L': 20,
                     u'M': 21,
                     u'N': 22,
                     u'NG': 23,
                     u'OW': 24,
                     u'OY': 25,
                     u'P': 26,
                     u'R': 27,
                     u'S': 28,
                     u'SH': 29,
                     u'T': 30,
                     u'TH': 31,
                     u'UH': 32,
                     u'UW': 33,
                     u'V': 34,
                     u'W': 35,
                     u'Y': 36,
                     u'Z': 37,
                     u'ZH': 38}
    return [phoneme_dict[phoneme] for phoneme in phoneme_list]

#converts a list of codes to the corresponding phonemes, including blanks
def code_to_phonemes(code_list):
    phoneme_dict =   {#' ': 0,
                     u'AA': 0,
                     u'AE': 1,
                     u'AH': 2,
                     u'AO': 3,
                     u'AW': 4,
                     u'AY': 5,
                     u'B': 6,
                     u'CH': 7,
                     u'D': 8,
                     u'DH': 9,
                     u'EH': 10,
                     u'ER': 11,
                     u'EY': 12,
                     u'F': 13,
                     u'G': 14,
                     u'HH': 15,
                     u'IH': 16,
                     u'IY': 17,
                     u'JH': 18,
                     u'K': 19,
                     u'L': 20,
                     u'M': 21,
                     u'N': 22,
                     u'NG': 23,
                     u'OW': 24,
                     u'OY': 25,
                     u'P': 26,
                     u'R': 27,
                     u'S': 28,
                     u'SH': 29,
                     u'T': 30,
                     u'TH': 31,
                     u'UH': 32,
                     u'UW': 33,
                     u'V': 34,
                     u'W': 35,
                     u'Y': 36,
                     u'Z': 37,
                     u'ZH': 38,
                        "-": 39}
    rev_dict = dict([(x[1], x[0]) for x in phoneme_dict.items()])
    return [rev_dict[i] for i in code_list]

#converts a list of codes to the corresponding phonemes, then groups repeats
def interpret_code(code_list):
    phoneme_list = code_to_phonemes(code_list)
    grouped_list = [k for (k, g) in itertools.groupby(phoneme_list)]
    return grouped_list

#given a list of prob distributions, returns a grouped list of phonemes.
def interpret_probs(prob_list, ignore_blanks = False):
    code_list = []

    #runs greedy on each prob distribution
    for prob_dist in prob_list:
        if ignore_blanks:
            prob_dist = prob_dist[:-1]

        guess = np.argmax(prob_dist)
        code_list.append(guess)

    return interpret_code(code_list)

'''
#given a label suffix, folder name, storage stem, and a reg expr, converts files
#matching the reg expr in batches of size batch_size.
#each batch is saved as storage_stem__batch_i.typename.npz.
#where typename is either stack or phoneme
#batch indices already in use are saved in storage_stem_inds.npy
#file names already converted are saved in stroage_stem.hist
def batch_convert_stack(label_suffix, data_folder_name, dest_folder_name, storage_stem, reg_expr, batch_size = 1000, extension = ".wav"):

    #loads up list of names
    hist_fname = storage_stem + '.hist'
    abs_hist_fname = os.path.join(dest_folder_name, hist_fname)

    #if dest folder doesn't exist, creates it
    if not os.path.isdir(dest_folder_name):
        os.makedirs(dest_folder_name)

    #same with history file
    if not os.path.isfile(abs_hist_fname):
        open(abs_hist_fname, 'a+').close()

    #figures out which inds are available
    batch_ind_expr = storage_stem +'\_batch\_(\d+)'
    used_batch_inds = map(int, get_names(dest_folder_name, batch_ind_expr))

    #makes sure batch ind is valid so nothing is overwritten
    if not used_batch_inds:
        current_batch_ind = 1
    else:
        current_batch_ind = max(used_batch_inds) + 1


    #makes sure already converted files are not converted again
    #opens hist file to append
    hist_file = open(abs_hist_fname, 'r')
    finished_names = [name.strip() for name in hist_file]
    hist_file.close()
    available_names = [name for name in get_names(data_folder_name, reg_expr) if name not in set(finished_names)]
    #starts converting files by batches
    for i in range(int(np.ceil(float(len(available_names))/batch_size))):
        batch_names = available_names[i*batch_size:(i+1)*batch_size]
        batch_dict = convert_to_dat_dict(label_suffix, data_folder_name, batch_names, extension)
        (batch_mfes, batch_phonemes) = zip(*(batch_dict.values()))
        batch_stacks = convert_to_stack(batch_mfes)

        #sorts out name to save batch as
        batch_name = storage_stem + '_batch_' + str(current_batch_ind)
        current_batch_ind += 1
        batch_stack_name = batch_name + '.stack'
        batch_phoneme_name = batch_name + '.phoneme'
        abs_stack_name = os.path.join(dest_folder_name, batch_stack_name)
        abs_phoneme_name = os.path.join(dest_folder_name, batch_phoneme_name)

        #saves batches
        np.savez(abs_stack_name, batch_stacks)
        np.savez(abs_phoneme_name, batch_phonemes)

        #adds names to history file
        hist_file = open(abs_hist_fname, 'a')
        hist_file.write('\n'.join(batch_names) + '\n')
        hist_file.close()
        print "Batch number " + str(current_batch_ind-1) + " is done."

    return True
'''
def clean_sentence(sentence):
    cleaned_sentence = map(lambda x : x if x in (string.ascii_lowercase + string.whitespace + r"'") else ' ', sentence.lower())
    return (" ".join("".join(cleaned_sentence).split()) + "\n")


#converts data in batches.
#uses multiprocessing
def batch_convert(label_suffix, data_folder_name, dest_folder_name, storage_stem
                 ,reg_expr, batch_size = 1000, extension = ".wav", num_threads = 1):

    #number of data points done in one point
    eff_batch_size = batch_size * num_threads

    #loads up list of names
    hist_fname = storage_stem + '.hist'
    abs_hist_fname = os.path.join(dest_folder_name, hist_fname)

    #if dest folder doesn't exist, creates it
    if not os.path.isdir(dest_folder_name):
        os.makedirs(dest_folder_name)

    #same with history file
    if not os.path.isfile(abs_hist_fname):
        open(abs_hist_fname, 'a+').close()

    #figures out which inds are available
    batch_ind_expr = storage_stem +'\_batch\_(\d+)'
    used_batch_inds = map(int, get_names(dest_folder_name, batch_ind_expr))

    #makes sure batch ind is valid so nothing is overwritten
    if not used_batch_inds:
        current_batch_ind = 1
    else:
        current_batch_ind = max(used_batch_inds) + 1

    #makes sure already converted files are not converted again
    #opens hist file to append
    print "Loading names..."
    hist_file = open(abs_hist_fname, 'r')
    finished_names = [name.strip() for name in hist_file]
    hist_file.close()
    available_names = [name for name in get_names(data_folder_name, reg_expr) if name not in set(finished_names)]
    print "Finished loading names."

    #helper function to get args
    get_args = lambda (ind, x) : (label_suffix, data_folder_name, x, extension
                                 , storage_stem, dest_folder_name, ind)

    #starts converting files by batches
    for i in range(int(np.ceil(float(len(available_names))/eff_batch_size))):
        batch_names = available_names[i*eff_batch_size:(i+1)*eff_batch_size]
        #gets the
        pooled_batch_names = [(current_batch_ind + j, batch_names[j*batch_size:(j+1)*batch_size]) for j in range(num_threads)]


        orig_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        #starts multiple pool objects
        pool = mp.Pool(processes = num_threads)
        signal.signal(signal.SIGINT, orig_sigint_handler)

        #does batch convert unless interrupted
        try:
            res = [pool.apply_async(single_batch_convert, get_args(batch_entry)) for batch_entry in pooled_batch_names]
            current_batch_ind += num_threads
            results = [p.get(360000) for p in res]
            pool.close()
            pool.join()
            [save_func(*(x+(abs_hist_fname,))) for x in results]

        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating processes"
            pool.terminate()
            pool.join()
            return False

    return True

#does the conversion for a single batch of data
def single_batch_convert(label_suffix, data_folder_name, batch_names, extension
                        , storage_stem, dest_folder_name, batch_ind):

    print "Batch number " + str(batch_ind) + " starting"
    batch_dict = convert_to_dat_dict(label_suffix, data_folder_name, batch_names, extension)
    (batch_mfes, batch_phonemes) = zip(*(batch_dict.values()))
    batch_ids = batch_dict.keys()


    #sorts out name to save batch as
    batch_name = storage_stem + '_batch_' + str(batch_ind)
    batch_mfe_name = batch_name + '.mfe'
    batch_phoneme_name = batch_name + '.phoneme'
    batch_id_name = batch_name + '.id'
    abs_mfe_name = os.path.join(dest_folder_name, batch_mfe_name)
    abs_phoneme_name = os.path.join(dest_folder_name, batch_phoneme_name)
    abs_id_name = os.path.join(dest_folder_name, batch_id_name)

    print "Batch number " + str(batch_ind) + " is done."
    return (abs_mfe_name, batch_mfes, abs_phoneme_name, batch_phonemes, abs_id_name, batch_ids, batch_names)


#helper function to save files, used with batch_convert
def save_func(mfe_loc, mfes, phoneme_loc, phonemes, id_loc, ids, names, hist_file_loc):
    np.savez(mfe_loc, mfes)
    np.savez(phoneme_loc, phonemes)
    hist_file = open(hist_file_loc, 'a')
    hist_file.write('\n'.join(names) + '\n')
    hist_file.close()
    id_file = open(id_loc, 'w+')
    id_file.write('\n'.join(ids) + '\n')
    id_file.close()

    return

#loads a single batch
def load_batch(batch_name):
    phoneme_name = batch_name + '.phoneme.npz'
    mfe_name = batch_name + '.mfe.npz'
    id_name = batch_name + '.id'
    letter_name = batch_name + '.letter.npz'
    f = open(id_name, 'r')
    ids = f.readlines()
    f.close()

    phoneme_dict = np.load(phoneme_name)
    phoneme_batch = phoneme_dict['arr_0']

    mfe_dict = np.load(mfe_name)
    mfe_batch = mfe_dict['arr_0']

    letter_dict = np.load(letter_name)
    letter_batch = letter_dict['arr_0']

    return (ids, mfe_batch, phoneme_batch, letter_batch)

#loads a single batch
def load_batch_no_letters(batch_name):
    phoneme_name = batch_name + '.phoneme.npz'
    mfe_name = batch_name + '.mfe.npz'
    id_name = batch_name + '.id'
    f = open(id_name, 'r')
    ids = f.readlines()
    f.close()

    phoneme_dict = np.load(phoneme_name)
    phoneme_batch = phoneme_dict['arr_0']

    mfe_dict = np.load(mfe_name)
    mfe_batch = mfe_dict['arr_0']


    return (ids, mfe_batch, phoneme_batch)

'''
#gives a generator given the batch size and a list of library names

def batch_generator_raw(batch_size, lib_fnames):
    lib_len = len(lib_fnames)
    (X_current, input_current, Y_current, label_current) = (np.array([]), np.array([]), np.array([]), np.array([]))
    lib_ind = 0
    while True:
        while Y_current.shape[0] < batch_size:
            if lib_ind == lib_len:
                lib_ind = 0
                eff_b_size = Y_current.shape[0]

                #the case when the last batch isn't of full size
                if eff_b_size != 0:
                    yield (X_current[:eff_b_size], input_current[:eff_b_size], Y_current[:eff_b_size], label_current[:eff_b_size])
                    X_current = X_current[eff_b_size:]
                    input_current = input_current[eff_b_size:]
                    Y_current = Y_current[eff_b_size:]
                    label_current = label_current[eff_b_size:]
            lib_dat = np.load(lib_fnames[lib_ind])
            lib_ind += 1
            if not Y_current.size:
                X_current = lib_dat['arr_1']
                input_current = lib_dat['arr_2']
                Y_current = lib_dat['arr_3']
                label_current = lib_dat['arr_4']
            else:
                X_current = np.concatenate((X_current, lib_dat['arr_1']), axis = 0)
                input_current = np.concatenate((input_current, lib_dat['arr_2']), axis = 0)
                Y_current = np.concatenate((Y_current, lib_dat['arr_3']), axis = 0)
                label_current = np.concatenate((label_current, lib_dat['arr_4']), axis = 0)

        yield (X_current[:batch_size], input_current[:batch_size], Y_current[:batch_size], label_current[:batch_size])
        X_current = X_current[batch_size:]
        input_current = input_current[batch_size:]
        Y_current = Y_current[batch_size:]
        label_current = label_current[batch_size:]

def batch_generator(batch_size, lib_fnames, n_phonemes, max_mfe_len):
    for (X, input_lengths, y, label_lengths) in batch_generator_raw(batch_size, lib_fnames):
        size_of_sample = len(input_lengths)
        inputs = {'log_mfes': X,
                  'the_labels': y,
                  'input_length': input_lengths,
                  'label_length': label_lengths,
                  }
        outputs = {'ctc': np.zeros([size_of_sample])
                  ,'phoneme_probs':np.zeros([size_of_sample,n_phonemes, max_mfe_len])}

        yield (inputs, outputs)
'''

'''
#gives a generator given the batch size and a list of library names
def batch_generator_raw(batch_size, lib_fnames, rand):
    lib_len = len(lib_fnames)
    (X_current, input_lens_current, Y_current, label_lens_current, ids_current) = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    shard_n = -1 # just to make the shard_n +=1 line work
    # Keep yielding batches forever
    while True:
        
        # First check if we have a shard and if there is enough data left in it
        if (not input_lens_current.size) or (not batch_start+batch_size <= input_lens_current.shape[0]):
            # If not: 
            
            # load a new shard using the current permutation, or create a new permutation if it's been used up
            shard_n += 1
            if shard_n % lib_len==0:
                shards_perm = np.random.permutation(lib_len) if rand else range(lib_len)
            shard_idx = shards_perm[shard_n%lib_len]
            new_shard = np.load(lib_fnames[shard_idx])
            
            # Mutate the current arrays and throw away the leftovers from the previous shard
            ids_current = new_shard['arr_0']
            X_current = new_shard['arr_1']
            input_lens_current = new_shard['arr_2']
            Y_current = new_shard['arr_3']
            label_lens_current = new_shard['arr_4']
            
            # set the batch start point to 0 and create a new permutation of examples
            batch_start = 0
            examples_perm = np.random.permutation(input_lens_current.shape[0]) if rand else range(input_lens_current.shape[0])
            
        # At this point there will definitely be enough data loaded
        batch_end = batch_start + batch_size
        batch_idxs = examples_perm[batch_start:batch_end]
        #print "X, ilens, Y, llens have shapes:", X_current[batch_idxs].shape, input_lens_current[batch_idxs].shape, Y_current[batch_idxs].shape, label_lens_current[batch_idxs].shape
        yield (X_current[batch_idxs], \
               input_lens_current[batch_idxs], \
               Y_current[batch_idxs], \
               label_lens_current[batch_idxs],
               ids_current[batch_idxs]
              )
        batch_start = batch_start+batch_size
        
def batch_generator(batch_size, lib_fnames, n_phonemes, max_mfe_len, rand=True):
    for (X, input_lengths, y, label_lengths, ids) in batch_generator_raw(batch_size, lib_fnames, rand):
        size_of_sample = len(input_lengths)
        inputs = {'log_mfes': X,
                  'the_labels': y,
                  'input_length': input_lengths,
                  'label_length': label_lengths,
                  }
        outputs = {'ctc': np.zeros([size_of_sample])
                  ,'phoneme_probs':np.zeros([size_of_sample,n_phonemes, max_mfe_len])}

        yield (inputs, outputs, ids)        
'''
def augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc_aug = np.roll(mfcc, r, axis=1)

    # zero padding
    if r > 0:
        mfcc_aug[:, :r] = 0
    elif r < 0:
        mfcc_aug[:, r:] = 0

    return mfcc_aug

#gives a generator given the batch size and a list of library names
def batch_generator(batch_size, set_name,n_phonemes, max_label_len, max_input_len, max_ratio, aug):
    # default data path
    _data_path = 'asset/data/'
    
    # Load the actual labels (not much data) and the mfcc paths (actual files are big)
    labels, mfcc_paths, ids = [], [], []
    csv_file =  open(_data_path + 'preprocess/meta/%s.csv' % set_name) 
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        # mfcc file
        mfcc_paths.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
        # label info ( convert to string object for variable-length support )
        labels.append(np.asarray(row[1:], dtype="int32"))
        ids.append(row[0])
    csv_file.close()
    
    n_examples = len(labels)
    
    # Iterate over the dataset forever
    while True:
        nonstandard_batch_count = 0
        # Generate a new permutation for each epoch
        perm = np.random.permutation(n_examples)
        # Iterate through the data using the permutation
        for batch_n in range(n_examples/batch_size):
            nonstandard_batch = False
            start = batch_n*batch_size
            end = (batch_n+1)*batch_size
            batch_idxs = perm[start:end]
            
            # Extract the labels and mfccs, ignoring extreme examples. 
            # Sometimes this results in a non-standard batch size, but that is okay
            batch_labels = []
            batch_mfccs = []
            batch_label_lens = []
            batch_input_lens = []
            batch_ids = []
            for i in batch_idxs:
                label = labels[i]
                mfcc = np.load(mfcc_paths[i])
                label_len = label.shape[0]
                input_len = mfcc.shape[0]
                len_ratio = float(input_len)/label_len
                if label_len <= max_label_len and input_len <= max_input_len and len_ratio <= max_ratio:
                    batch_labels.append(label)
                    mfcc_aug = augment_speech(mfcc) if aug else mfcc
                    batch_mfccs.append(mfcc_aug)
                    batch_label_lens.append(label_len)
                    batch_input_lens.append(input_len)
                    batch_ids.append(ids[i])
                else:
                    continue
                    nonstandard_batch = True
            if nonstandard_batch:
                nonstandard_batch_count += 1
            actual_batch_size = len(batch_ids)
            batch_y = keras.preprocessing.sequence.pad_sequences(batch_labels, 
                                                           maxlen=max_label_len, 
                                                           dtype='int32',
                                                           padding='post', 
                                                           truncating='post', 
                                                           value=-1)
            batch_X = keras.preprocessing.sequence.pad_sequences(batch_mfccs, 
                                                           maxlen=max_input_len, 
                                                           dtype='float32',
                                                           padding='post', 
                                                           truncating='post', 
                                                           value=0.)
            batch_inputs = {'mfccs': batch_X,
                  'labels': batch_y,
                  'input_lens': np.asarray(batch_input_lens),
                  'label_lens': np.asarray(batch_label_lens)}
            batch_outputs = {'ctc': np.zeros([actual_batch_size]),
                       'phoneme_probs':np.zeros([actual_batch_size,n_phonemes, max_input_len])}
            yield (batch_inputs, batch_outputs)#, batch_ids)
            
def get_test_data(n_phonemes, max_label_len, max_input_len):
    # default data path
    _data_path = 'asset/data/'
    
    # Load the actual labels (not much data) and the mfcc paths (actual files are big)
    labels, mfccs, ids, input_lens, label_lens = [], [], [], [], []
    csv_file =  open(_data_path + 'preprocess/meta/test.csv') 
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        # mfcc file
        mfcc = np.load(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
        
        label = np.asarray(row[1:], dtype="int32")
        # label info ( convert to string object for variable-length support )
        if mfcc.shape[0] <= max_input_len and label.shape[0] <= max_label_len:
            labels.append(label)
            input_lens.append(min(500,mfcc.shape[0]))
            mfccs.append(mfcc)
            label_lens.append(label.shape[0])
            ids.append(row[0])
       
    csv_file.close()
    
    n_examples = len(labels)
    
    y = keras.preprocessing.sequence.pad_sequences(labels, 
                                                   maxlen=max_label_len, 
                                                   dtype='int32',
                                                   padding='post', 
                                                   truncating='post', 
                                                   value=-1)
    X = keras.preprocessing.sequence.pad_sequences(mfccs, 
                                                   maxlen=max_input_len, 
                                                   dtype='float32',
                                                   padding='post', 
                                                   truncating='post', 
                                                   value=0.)
    test_inputs = {'mfccs': X,
                   'labels': y,
                   'input_lens': np.asarray(input_lens),
                   'label_lens': np.asarray(label_lens)}
    test_outputs = {'ctc': np.zeros([n_examples]),
                    'phoneme_probs':np.zeros([n_examples,n_phonemes, max_input_len])}
    return (test_inputs, test_outputs, ids)
'''            
def batch_generator(batch_size, lib_fnames, n_phonemes, max_mfe_len, rand=True):
    for (X, input_lens, y, label_lens, ids) in batch_generator_raw(batch_size, lib_fnames, rand):
        size_of_sample = len(input_lengths)
        inputs = {'log_mfes': X,
                  'the_labels': y,
                  'input_length': input_lens,
                  'label_length': label_lens,
                  }
        outputs = {'ctc': np.zeros([size_of_sample])
                  ,'phoneme_probs':np.zeros([size_of_sample,n_phonemes, max_mfe_len])}

        yield (inputs, outputs, ids)  
'''
def phonemes_to_letters(phoneme_list, reversed_dict):
    space_inds = [i for i, x in enumerate(phoneme_list) if x == ' ']
    grouped_phoneme_list = split_at_inds(space_inds, phoneme_list)
    return [reversed_dict[x] for x in grouped_phoneme_list]

def split_at_inds(inds, src_list):
    grouped_list = []
    start_ind = 0
    for index in inds:
        grouped_list.append(",".join(src_list[start_ind:index]))
        start_ind = index + 1
    return grouped_list

def batch_p_to_l(batch_name, reversed_dict):
    phoneme_name = batch_name + '.phoneme.npz'
    phoneme_dict = np.load(phoneme_name)
    phoneme_batch = phoneme_dict['arr_0']
    letter_batch = [phonemes_to_letters(x, reversed_dict) for x in phoneme_batch]
    letters_name = batch_name + '.letter'
    np.savez(letters_name, letter_batch)

def batch_generator_chars(batch_size, lib_fnames, max_mfe_len, n_letters = 28):
    for (X, input_lengths, y, label_lengths) in batch_generator_raw(batch_size, lib_fnames):
        size_of_sample = len(input_lengths)
        inputs = {'log_mfes': X,
                  'the_labels': y,
                  'input_length': input_lengths,
                  'label_length': label_lengths,
                  }
        outputs = {'ctc': np.zeros([size_of_sample])
                  ,'letter_probs':np.zeros([size_of_sample,n_letters, max_mfe_len])}

        yield (inputs, outputs)


        
def ids_to_letters(batch_name, data_folder, suffix = ".flac.trn" ):
    id_name = batch_name + '.id'
    f = open(id_name, 'r')
    ids = [x.rstrip('\n') for x in f.readlines()]
    f.close()
    abs_file_names = [os.path.join(data_folder, idx + suffix) for idx in ids]
    batch_letters = []
    for fname in abs_file_names:
        fid = open(fname, 'r')
        text = fid.read()
        fid.close()
        batch_letters.append(list(text))

    batch_letter_name = batch_name + '.letter'
    np.savez(batch_letter_name, batch_letters)
    return True
