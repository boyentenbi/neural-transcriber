# neural-transcriber

Our (mine and Akuan's) goal was to do transcription on-device and in real time. We made the neural acoustic model very easily, but we ran out of time trying to get the language model working. 

Contains code to preprocess TedLium, Librispeech and VCTK from https://github.com/buriburisuri/speech-to-text-wavenet. We'd have used the Switchboard corpus if we had £1500 to spare.

The peaks of our model's softmax layer on me saying 'What is the phone number?':
![peaks](https://github.com/boyentenbi/neural-transcriber/blob/master/phonemectc.png)


# Stuff I learned

* How to use terminal multiplexing (very useful)
* How to do 'pipelining'. Converting MP3s to MFCCs takes a long time and we had to robustly convert gigabytes of data on spot-instances which are prone to failure
* How finite automata are useful in ASR
* A bit about profiling code
* Training the neural net is almost always the easy bit. Preprocessing and post-processing are a lot harder.

# Details

We achieved alignment-free phoneme and character recognition using a neural convolution through time on the MFCC representation of an audio signal. 

We tried the following combinations of these with language models:

1. Character-level acoustic model including a 'space' character. No language model.
2. Character-level acoustic model inlcuding space. WFST language model.
3. Phoneme-level acoustic model and from-scratch beam search language model.
4. Phoneme-level acoustic model and WFST language model.

All but the first were about 2 orders of magnitude slower than https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41176.pdf

From profiling, we know this was due to our language models. However, we ran out of time before we could fully diagnose the problem.

The light-blue line represents the 'blank' symbol (see ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)

