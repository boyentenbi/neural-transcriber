import MultiTree as MT
from copy import deepcopy
import numpy as np

class DecodeInfo:
    def __init__(self, w_h = [], p_w = [], orig = None):
        if orig == None:
            self.word_hist = deepcopy(w_h)
            self.partial_word = deepcopy(p_w)
            self.tree = MT.search_tree
        else:
            self.word_hist = deepcopy(orig.word_hist)
            self.partial_word = deepcopy(orig.partial_word)
            self.tree = orig.tree
    def __repr__(self):
        return " ".join(self.word_hist) + "|" + " ".join(self.partial_word)
    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, DecodeInfo):
            return (self.word_hist, self.partial_word) == (other.word_hist, other.partial_word)
        else:
            return False
    
    def empty(self):
        return not (self.word_hist or self.partial_word)
    
    def last_phoneme(self):
        if self.partial_word:
            return self.partial_word[-1]
        
        elif self.word_hist:
            #gives last phoneme of last word
            last_word = self.word_hist[-1]
            
            #arpabet is word to phoneme dictionary, stresses removed
            return MT.arpabet[last_word][-1]
        else:
            return None
        
            
    def one_fewer(self):
        if self.partial_word:
            return DecodeInfo(w_h = self.word_hist, p_w = self.partial_word[:-1])
        elif self.word_hist:
            last_word = self.word_hist[-1]
            last_phons = MT.arpabet[last_word]
            return DecodeInfo(w_h = self.word_hist[:-1], p_w = self.partial_word[:-1])
        else:
            print "Error: no hat version of empty word"
            return None
    
    def add_phon(self, new_phon):
        self.partial_word.append(new_phon)
        self.tree = self.tree.lookup([new_phon])
        if self.tree == None:
            return False
        
        else:
            return True

def times(*args):
    if isinstance(args[0], list):
        mult_list = args[0]
    else:
        mult_list = [arg for arg in args]
    
    if 0. in mult_list:
        return 0
    else:
        return np.exp(np.sum(np.asarray([np.log(x) for x in mult_list])))
    
def power(base, exponent):
    if base <= 0.:
        return 0.
    return np.exp(exponent*np.log(base))

def divide(num, denom):
    if num == 0.:
        return 0.
    return np.exp(np.log(num) - np.log(denom))

def get_best_paths(B, pr_b, pr_nb, width, time):
    def get_prob(y):
        return pr_b(time, y) + pr_nb(time, y)
    
    return sorted(list(set(B)), key=get_prob, reverse=True)[:width]


def beam_search(T, p_ctc, p_lang, width, beta):

    pr_nb_store = [{}]*(T+1)
    pr_b_store = [{}]*(T+1)
    
    #extension probability of adding a phoneme k
    def p_ext(k, y, t):
        p_last = pr_nb(t-1, y)
        if y.last_phoneme() != k:
            p_last += pr_b(t-1, y)
        return times([p_ctc(k,t), power(p_trans(k, y), beta), p_last])
    
    #derived phoneme transition prob from language model
    def p_trans(k, y):
        if y.tree.lookup([k]) == None:
            return 0.
        
        denom = np.sum(np.asarray([p_lang(x, y) for x in y.tree.get_leaves()]))
        numer = np.sum(np.asarray([p_lang(x, y) for x in y.tree.lookup([k]).get_leaves()]))
        return divide(numer, denom)
    
    #penalty probability for converting to a homophone
    def p_homophone(word, y):
        denom = np.sum(np.asarray([p_lang(x, y) for x in y.tree.get_leaves()]))
        numer = p_lang(word, y)
        return divide(numer, denom)
    #initialising non-blank prob
    empty_str = DecodeInfo()
    pr_nb_store[0][empty_str] = 1.
    pr_b_store[0][empty_str] = 0.
    
    def pr_nb(t, y):
        if y in pr_nb_store[t]:
            return pr_nb_store[t][y]
        else:
            return 0.
    
    def pr_b(t, y):
        if y in pr_b_store[t]:
            return pr_b_store[t][y]
        else:
            return 0.
    
    #most probable sequences
    B = [empty_str]
    
    for t in range(1, T+1):
            B_hat = get_best_paths(B, pr_b, pr_nb, width, t-1)
            print "\n".join(str(elem) for elem in B_hat) + "\n\n"
            B = []
            for y in B_hat:
                if not y.empty():
                    pr_nb_store[t][y]= times([pr_nb(t-1, y), p_ctc(y.last_phoneme(), t)])
                    if y.one_fewer() in B_hat:
                        pr_nb_store[t][y] += p_ext(y.last_phoneme(), y.one_fewer(), t)

                pr_b_store[t][y] = times(pr_b(t-1,y), p_ctc("-", t))

                B.append(y)
                for k in MT.phonemes:

                    #if adding a phoneme k is valid
                    if k in y.tree.branches:

                        pr_plus = p_ext(k, y, t)

                        #initialising y_new = y+k
                        y_new = DecodeInfo(orig = y)
                        y_new.add_phon(k)

                        conversion_penalty_prob = 0.
                        #handles conversion from phoneme string to word
                        for word in y_new.tree.leaf:

                            #converted_y converts phoneme part of y_new into a new word
                            converted_y = DecodeInfo()
                            converted_y.word_hist = deepcopy(y_new.word_hist)
                            converted_y.word_hist.append(word)
                            pr_b_store[t][converted_y] = 0.
                            pr_nb_store[t][converted_y] = times(pr_plus, power(p_homophone(word, y_new), beta))
                            B.append(converted_y)
                            conversion_penalty_prob += p_homophone(word, y_new)

                        #penalises non-converted string
                        pr_nb_store[t][y_new] = times(pr_plus, power(conversion_penalty_prob, beta))
                        B.append(y_new)

    return (B, pr_b_store, pr_nb_store)#get_best_paths(B, pr_b_store, pr_nb_store, 1)
