import nltk
from copy import deepcopy
import string

#gets word level n-grams
def get_n_grams(sentence, gram_count, max_n, null_char = "@"):
    for n in range(1, max_n+1):
        
        #adds null chars at the start of the sentence
        n_grams = [" ".join(x) for x in nltk.ngrams((null_char + " ") * (n-1) + clean_sentence(sentence), n)]
        for gram in n_grams:
            if gram in gram_count:
                gram_count[gram] += 1
            
    

def clean_sentence(sentence):
    cleaned_sentence = map(lambda x : x if x in (string.ascii_lowercase + string.whitespace + r"'") else ' ', sentence.lower())
    return (" ".join("".join(cleaned_sentence).split()) + "\n")

def gen_poss_grams(word_list, max_n):
    poss_keys = deepcopy(word_list)
    for i in range(max_n-1):
        new_keys = []
        for key in poss_keys:
            new_keys.extend([(key + x) for x in ([(" " + word) for word in word_list] + [""])])

        poss_keys.extend(new_keys)

    return {key: 0 for key in poss_keys}
