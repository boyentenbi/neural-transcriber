import numpy as np
import pandas as pd
import glob
import csv
import librosa
import scikits.audiolab
import os
import subprocess
import nltk
from process_data import code_label
# data path
_data_path = "asset/data/"


#
# process VCTK corpus
#
D = nltk.corpus.cmudict.dict()
def process_vctk(csv_file):
    
    print "Starting VCTK preprocessing.."

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read label-info
    df = pd.read_table(_data_path + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                       index_col=False, delim_whitespace=True)

    # read file IDs
    file_ids = []
    for d in [_data_path + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])
    print "Got {} file ids".format(len(file_ids))
    VCTK_counter = 0
    no_label_counter = 0
    exist_counter = 0
    for i, f in enumerate(file_ids):

        # wave file name
        wave_file = _data_path + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            exist_counter+=1
            continue
        
        # print info
        #print("VCTK corpus preprocessing (%d / %d) - '%s']" % (i, len(file_ids), wave_file))
        
        # get label index
        uncleaned_label = open(_data_path + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt').read()
        coded_label = code_label(uncleaned_label, D)
        if not coded_label:
            no_label_counter += 1
            continue
        # load wave file
        wave, sr = librosa.load(wave_file, mono=True, sr=None)

        # re-sample ( 48K -> 16K )
        wave = wave[::3]

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000).T # transpose because librosa returns with time axis last instead of first

        
        
        # save result ( exclude small mfcc data to prevent ctc loss )
        if coded_label and len(coded_label) < mfcc.shape[0]:
            # save meta info
            writer.writerow([fn] + coded_label)
            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)
            VCTK_counter +=1
    print "Finished processing VCTK. {} new examples, {} already exist and {} un-labellable".format(VCTK_counter, exist_counter, no_label_counter)


#
# process LibriSpeech corpus
#

def process_libri(csv_file, category):
    print "Starting LibriSpeech/{} preprocessing...".format(category)
    
    parent_path = _data_path + 'LibriSpeech/' + category + '/'
    labels, wave_files = [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        uncleaned_label = ' '.join(field[1:])
                        labels.append(code_label(uncleaned_label, D))
    print "Got {} files in LibriSpeech/{}".format(len(labels), category)
    # save results
    libri_counter = 0
    no_label_counter = 0
    exist_counter = 0
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists(target_filename):
            exist_counter+=1
            continue
        if not label:
            no_label_counter += 1
            continue
        # print info
        #print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        wave, sr, _ = scikits.audiolab.flacread(wave_file)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000).T

        # save result ( exclude small mfcc data to prevent ctc loss )
        if label and len(label) < mfcc.shape[0]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)
            libri_counter += 1
    print "Finished processing LibriSpeech/{}. {} new examples, {} already exist and {} un-labellable".format(category, libri_counter, exist_counter, no_label_counter )

#
# process TEDLIUM corpus
#
def convert_sph( sph, wav ):
    """Convert an sph file into wav format for further processing"""
    command = [
        'sox','-t','sph', sph, '-b','16','-t','wav', wav
    ]
    subprocess.check_call( command ) # Did you install sox (apt-get install sox)

def process_ted(csv_file, category):
    print "Processing TEDLIUM corpus..."
    parent_path = _data_path + 'TEDLIUM_release2/' + category + '/'
    labels, wave_files, offsets, durs = [], [], [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read STM file list
    stm_list = glob.glob(parent_path + 'stm/*')
    for stm in stm_list:
        with open(stm, 'rt') as f:
            records = f.readlines()
            for record in records:
                field = record.split()

                # wave file name
                wave_file = parent_path + 'sph/%s.sph.wav' % field[0]
                wave_files.append(wave_file)

                # label index
                uncleaned_label = ' '.join(field[6:])
                labels.append(code_label(uncleaned_label, D))

                # start, end info
                start, end = float(field[3]), float(field[4])
                offsets.append(start)
                durs.append(end - start)
    print "Got {} files in TEDLIUM/{}".format(len(labels), category)
    # save results
    ted_counter = 0
    no_label_counter = 0
    exist_counter = 0
    for i, (wave_file, label, offset, dur) in enumerate(zip(wave_files, labels, offsets, durs)):
        fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            exist_counter+=1
            continue
        if not label:
            no_label_counter += 1
            continue
        # print info
        #print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files), wave_file, offset))
        # load wave file
        if not os.path.exists( wave_file ):
            sph_file = wave_file.rsplit('.',1)[0]
            if os.path.exists( sph_file ):
                convert_sph( sph_file, wave_file )
            else:
                raise RuntimeError("Missing sph file from TedLium corpus at %s"%(sph_file))
        wave, sr = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000).T

        # save result ( exclude small mfcc data to prevent ctc loss )
        if label and len(label) < mfcc.shape[0]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)
            ted_counter +=1
    print "Finished processing TEDLIUM/{}. {} new examples, {} already exist and {} un-labellable".format(category, ted_counter, exist_counter, no_label_counter)

#
# Create directories
#

if not os.path.exists('asset/data/preprocess'):
    os.makedirs('asset/data/preprocess')
if not os.path.exists('asset/data/preprocess/meta'):
    os.makedirs('asset/data/preprocess/meta')
if not os.path.exists('asset/data/preprocess/mfcc'):
    os.makedirs('asset/data/preprocess/mfcc')


#
# Run pre-processing for training
#

# VCTK corpus
csv_f = open('asset/data/preprocess/meta/train.csv', 'a+')
process_vctk(csv_f)
csv_f.close()

csv_f = open('asset/data/preprocess/meta/train.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "train.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()
                                               
# LibriSpeech corpus for train
csv_f = open('asset/data/preprocess/meta/train.csv', 'a+')
process_libri(csv_f, 'train-clean-360')
#process_libri(csv_f, 'train-clean-100')
#process_libri(csv_f, 'train-other-500')
csv_f.close()

csv_f = open('asset/data/preprocess/meta/train.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "train.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()

# TEDLIUM corpus for train
csv_f = open('asset/data/preprocess/meta/train.csv', 'a+')
process_ted(csv_f, 'train')
csv_f.close()
                                               
csv_f = open('asset/data/preprocess/meta/train.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "train.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()                                               

#
# Run pre-processing for validation
#

# LibriSpeech corpus for valid
csv_f = open('asset/data/preprocess/meta/valid.csv', 'a+')
process_libri(csv_f, 'dev-clean')
csv_f.close()

csv_f = open('asset/data/preprocess/meta/valid.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "valid.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()   

# TEDLIUM corpus for valid
csv_f = open('asset/data/preprocess/meta/valid.csv', 'a+')
process_ted(csv_f, 'dev')
csv_f.close()

csv_f = open('asset/data/preprocess/meta/valid.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "valid.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()   

#
# Run pre-processing for testing
#

# LibriSpeech corpus for test
csv_f = open('asset/data/preprocess/meta/test.csv', 'a+')
process_libri(csv_f, 'test-clean')
csv_f.close()

csv_f = open('asset/data/preprocess/meta/test.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "test.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()   

# TEDLIUM corpus for test
csv_f = open('asset/data/preprocess/meta/test.csv', 'a+')
process_ted(csv_f, 'test')
csv_f.close()

csv_f = open('asset/data/preprocess/meta/test.csv', 'r')
reader = csv.reader(csv_f, delimiter=",")
print "test.csv has {} rows currently".format(len([x for x in reader]))
csv_f.close()   


# Run preprocessing tests
print "Running tests on processed data..."

mfcc_paths = os.listdir("asset/data/preprocess/mfcc")
n_mfccs=len(mfcc_paths)
print "There are {} saved mfccs".format(n_mfccs)
perm = np.random.permutation(n_mfccs)
lens = [np.load("asset/data/preprocess/mfcc/"+mfcc_paths[i]).shape[0] for i in perm[:min(10000,n_mfccs)]]
#tenth = np.percentile(lens, 10)
mfcc_ninetieth = np.percentile(lens, 95)
print "Random sample of mfccs has 95% of mfccs with length less than {}".format(mfcc_ninetieth)


label_lens = []

for set_name in ["train", "valid", "test"]:
#     set_name = "train"
    # Get the labels and 
    csv_file =  open('asset/data/preprocess/meta/{}.csv'.format(set_name), "r") 
    reader = csv.reader(csv_file, delimiter=',')
    set_lens = [len(row)-1 for row in reader]
    print "There are {} labels in {}.csv".format(len(set_lens), set_name) 
    label_lens += set_lens
    csv_file.close()
n_labels = len(label_lens)
print "There are {} labels in total".format(n_labels) 
label_ninetieth = np.percentile(label_lens, 95)
print "Over the whole dataset, 95% of labels have length less than {}".format(label_ninetieth)

rows = []
for set_name in ["train", "valid", "test"]:
    csv_file = open('asset/data/preprocess/meta/{}.csv'.format(set_name), "r") 
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        rows.append(row)
    csv_file.close()
    
perm = np.random.permutation(n_labels)

sample_rows = [rows[i] for i in perm[:min(10000, len(rows))]]

len_ratios = []
for row in sample_rows:
    filename = row[0]
    input_len = np.load("asset/data/preprocess/mfcc/"+filename+".npy").shape[0]
    label_len = len(row[1:])
    ratio = float(input_len)/label_len
    len_ratios.append(ratio)
ratios_ninetieth = np.percentile(len_ratios, 95)
print "Random sample has 95% of examples with a input_length to label_length ratio of less than {}".format(ratios_ninetieth)
