import nltk
import numpy as np

def get_n_grams(sentence, vocabulary, max_n):
    for n in range(1, max_n+1):
        n_grams = [str(x) for x in nltk.ngrams(clean_sentence(sentence), n)]
        for gram in n_grams:
            if gram in vocabulary:
                vocabulary[gram] += 1
            else:
                vocabulary[gram] = 1

def clean_sentence(sentence):
    return [c for c in sentence.lower() if c.isalpha() or c == " "]

def gen_vocab(alphabet, max_n):
    poss_keys = list(alphabet)
    for i in range(max_n-1):
        new_keys = []
        for key in poss_keys:
            new_keys.extend([key + x for x in list(alphabet) + [""]])

        poss_keys.extend(new_keys)

    return {key: 0 for key in poss_keys}

def count_file(file_name, gram_dict, max_n):
    f = open(file_name, "r")
    for (i, sentence) in enumerate(f.readlines()):
        get_n_grams(sentence, gram_dict, max_n)
        if i % 100000 == 0:
            print "Have counted {} lines from file {}".format(i+1, file_name)

    np.save(file_name + "_count.npy", gram_dict)
    print "Have saved count for file {}".format(file_name)
    f.close()
    return True

def stimes(factors):
    if 0. in factors:
        return 0.
    else:
        return np.exp(np.sum(np.log(factors)))
    
def sexp(base, exp):
    if exp == 0:
        return base
    return np.exp(np.log(base) * exp)
    
def top_keys(p_string, k, beta):
    (strings, p_list) = zip(*p_string.items())
    p_arr = np.asarray(p_list)
    string_lens = np.asarray([len(string) for string in strings])
    len_multipliers = string_lens ** beta
    top_k_idxs = np.argsort(p_arr * len_multipliers)[-k:] 
    top_k_keys = [strings[i] for i in top_k_idxs]
    return top_k_keys

#does a beam search with width n_paths
def beam_search(T, n_paths, alpha, beta, prob_char, prob_lm, chars = "abcdefghijklmnopqrstuvwxyz -"):

    Z = [[] for t in range(T+1)]
    p_s_b = [{} for t in range(T+1)]
    p_s_nb = [{} for t in range(T+1)]
    p_s = [{} for t in range(T+100)]

    # Just an initialization that works...
    p_s_nb[0] = {"":1.}
    p_s_b[0] = {"": 0.}

    p_s[0] = {"":1.}
    Z[0] = [""]

    
    # Iterate over input timesteps
    for t in range(1, T+1):

        # Consider all of the strings which were candidates given data up to time t-1
        # and compute the probabilities of 'some' new strings given data up to time t
        # 'some' = the current strings and 1-character extensions of them
        for string in Z[t-1]:

            # Probs for existing candidates when the final model out is blank, and when it is not blank
            p_s_b[t][string] = stimes( [p_s[t-1][string] , prob_char(t, "-")])
            p_s_nb[t][string] = 0. if string == "" else stimes([p_s_nb[t-1][string], prob_char(t, string[-1])])

            # Now consider extensions of one character
            for c in chars[:-1]:
                # When extension is not the same as final character
                if not string or c != string[-1]:
                    p_s_nb[t][string+c] = stimes([prob_char(t, c) , p_s[t-1][string] , sexp(prob_lm(string, c), alpha)])
                    p_s_b[t][string+c] = 0.
                # When extension is the same as final character
                else:
                    p_s_nb[t][string+c] = stimes([prob_char(t, c), p_s_b[t-1][string] , sexp(prob_lm(string, c), alpha)]) 
                    p_s_b[t][string+c] = 0.
            # Set the probability of a string given input data up to time t as the sum of the marginals over t'th model out
            p_s[t] = {s: p_s_b[t][s] + p_s_nb[t][s] for s in p_s_nb[t].keys()}
            
            # Take the top k strings 
            Z[t] = top_keys(p_s[t], n_paths, beta)
            
    return (Z, p_s)