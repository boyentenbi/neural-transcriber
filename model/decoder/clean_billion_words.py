import os
import re
import nltk
import string
from n_gram_words import clean_sentence
import inflect
p = inflect.engine()


folder_name = "language_model_dat/1-billion-word/training"
f_names = os.listdir(folder_name)
text_names = [os.path.join(folder_name, f_name) for f_name in f_names if re.match(r"^news.en-([0-9]+)-of-00100$", f_name)]

def convert_number(number):
    a = float(number)
    if int(a) == a and a <= 1999 and a>=1500:
        return convert_number(str(number)[:2]) + " " + convert_number(str(number)[2:])
    return re.sub(r"-", " ", p.number_to_words(number))
    
bad_punc = [s for s in string.punctuation if not s in "'"]
def clean_sentence(sentence):
    x = sentence.lower()
    x = re.sub(r"([-+]?\d*\.\d+|\d+)", lambda z: convert_number(z.group()), x) 
    x = re.sub(r"'", "", x) 
    x = [c if not c in bad_punc else ' ' for c in x]
    x_list = "".join(x).split()
    
#     cleaned_sentence = map(lambda x : x if x in allowed_chars else ' ', sentence.lower())
    return (" ".join(x_list) + "\n")
print "Starting cleaning of billion words..."
f = open("language_model_dat/processed-text-clean-v2.txt", "w")
# chars_to_remove = [x for x in string.punctuation if not x in "."]
# table = string.maketrans(" "*len(chars_to_remove), )
for (i,fname) in enumerate(text_names):
    t_f = open(fname, "r")
    file_lines = t_f.readlines()
    for line in file_lines:
        try:
            cleaned = clean_sentence(line)
            f.write(cleaned)
        except:
            print "failed to clean the following sentence:"
            print line
        
    
    print "Done with text {}".format(i)
    #print cleaned # the last one of this text
#     break
f.close()
print "Done converting all texts!"