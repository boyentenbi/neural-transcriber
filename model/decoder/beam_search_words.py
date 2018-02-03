n_paths = 5
alpha = 1
beta = 0.1

phons = #"abcdefghijklmnopqrstuvwxyz -"

alphabet = dict(zip(chars, range(28) ))
rev_alphabet = dict(zip(range(28), (chars)))
T = char_probs.shape[0]

Z = [[] for t in range(T)]
p_b = [{} for t in range(T)]
p_nb = [{} for t in range(T)]
p = [{} for t in range(T)]
p_w_p = [{} for t in range(T)]
# Just an initialization that works...
p_nb[0] = {("",""):1.}
p_b[0] = {("",""): 0.}

p[0] = {("",""):1.}
Z[0] = [("", "")]

D = nltk.corpus.cmudict.dict()
rev_dict = {v[0] : k for (k, v) in D.items()}

def stimes(factors):
    return np.exp(np.sum(np.log(factors)))
   
def sexp(base, exp):
    return np.exp(np.log(base) * exp)
def phonemes_to_homophones(phonemes):
    phoneme_list = phonemes.split(" ")

    
def top_keys(d, k, beta):
    sentences = [a[0] for a in d.keys()]
    logprobs = np.asarray(d.values())
    string_lens = np.asarray([len(sentence)+1 for sentence in sentences], dtype=float)
    
    top_k_idxs = argsort(logprobs / string_len)[-k:] # dividing makes them greater because they're negative!
    top_k_keys = [d.keys()[i] for i in top_k_idxs]
    return top_k_keys

def beam_search():
    # Iterate over input timesteps
    for t in range(1, T+1):

        # Consider all of the (w,p) pairs which were candidates given data up to time t-1
        # and compute the probabilities of 'some' new strings given data up to time t
        # 'some' = the current strings and 1-phoneme extensions of them
        for (w,p) in Z[t-1]:
            
            # We need to compute:
            
            # 1. p_nb[t][(w,p)]
            # 2. p_b[t][(w,p)]
            # 3. p_nb[t][(w,p + k)] for all k not blank. Remembering that this may lead to a new word.
           
            # Consider the extension by a blank phoneme
            p_nb[t][(w,p)] = p_nb[t-1][(w,p)] + char_probs[t, alphabet[(w,p)[-1]]] if (w,p)!=("","") else None # plus some other stuff???
            p_b[t][(w,p)] =  p[t-1][(w,p)] + char_probs[t, alphabet["-"]]
        
            # Consider extensions of the uncontracted thing by one phoneme
            for k in phons[:-1]:
                
                # When the extension definitely adds a phoneme to the contraction
                # all the probs are nb because we aren't adding blank!
                if k!="-" and k != (w,p)[-1]:

                    # if p+k can be read as a word v, compute the probability of (wv,0) using the lm
                    # for every such v
                    # Note that (-p_w_p_star[(w,p)])*alpha exactly cancels the same term in a prev computation of p[t-1][(w,p)]
                    # This is because (unlogarithmed) p[t-1] can be factored into acoustic and language factors
                    # where exp(p_w_p_star[(w,p)]*alpha) is the language factor
                    # The new language model factor is just the lm evaluated on v and the sentence
                    for h in phonemes_to_homophones(p+" "+k):
                        p_nb[t][(w+" "+h, "")] = char_probs[t, alphabet[k]] \
                                                    + p[t-1][(w,p)]   \
                                                    + (lm(h, w)-p_w_p_star[(w,p)]) * alpha
                        
                        p_b[t][(w+" "+h, "")] =  None
                              
                    # Compute the probability of (w,p+k) under the language model, NOT CONSIDERING THE ACOUSTIC MODEL
                    # This is the sum of probabilities of wx where p is a strict prefix of every x
                    p_w_p_star[(w, p+k)] = get_p_w_p_star(w, p+k)
                            
                    p_nb[t][(w,p+k)] = char_probs[t, alphabet[k]] \
                                        + p[t-1][(w,p)]   \
                                        + (p_w_p_star[(w,p+k)]-p_w_p_star[(w,p)]) * alpha
                            
                # When extension is the same as the final phoneme
                # the extension only extends y if the final phoneme in the uncontracted thing is blank
                elif k!="-" and k==(w,p)[-1]:
                    p_nb[t][(w,p+k)] = char_probs[t, alphabet[k]] \
                                        + p_b[t-1][(w,p)] \
                                        + (p_w_p_star[(w,p+k)]-p_w_p_star[(w,p)]) * alpha 
                else: # k == "-"
                    continue
                
            # Set the probability of a string given input data up to time t as the sum of the marginals over t'th model out
            p[t] = {y: np.log(np.exp(p_b[t][y]) + np.exp(p_nb[t][y])) for y in p_b[t].keys()}
            # Take the top k strings 
            Z[t] = top_keys(p[t], n_paths, beta)
    return Z[T]
    # Don't forget to throw away the p's when you're done!