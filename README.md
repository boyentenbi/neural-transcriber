# Transcriber

Mine and Akuan's neural + WFST transcriber.

Contains code to preprocess TedLium, Librispeech and VCTK from https://github.com/buriburisuri/speech-to-text-wavenet.

Achieves alignment-free transcription using a neural convolution through time on the MFCC representation of an audio signal.

We tried the following configurations:

1. Character-level acoustic model including a 'space' character. No language model.
2. Character-level acoustic model inlcuding space. WFST language model.
3. Phoneme-level acoustic model and from-scratch beam search language model.
4. Phoneme-level acoustic model and WFST language model.

All but the first were about 2 orders of magnitude slower than https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41176.pdf

Softmax peaks on 'What is the phone number?':
![peaks](https://github.com/boyentenbi/neural-transcriber/blob/master/phonemectc.png)

The light-blue line represents the 'blank' symbol (see ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf)

