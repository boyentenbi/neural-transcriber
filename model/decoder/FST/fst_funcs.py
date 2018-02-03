import pywrapfst as pfst
import numpy as np
import nltk 

phons = [u'AA',u'AE',u'AH',u'AO',u'AW',u'AY',u'B',u'CH',
u'D',u'DH',u'EH',u'ER',u'EY',u'F',u'G',u'HH',u'IH',u'IY',u'JH',
u'K',u'L',u'M',u'N',u'NG',u'OW',u'OY',u'P',u'R',u'S',u'SH',u'T',
u'TH',u'UH', u'UW',u'V',u'W',u'Y',u'Z',u'ZH', u'-']

def analyse_state(fst, state):
    targets = {}
    for k in fst.arcs(state):
        if k.nextstate in targets.keys():
            targets[k.nextstate].append(fst.input_symbols().find(k.ilabel))
        else:
            targets[k.nextstate]=[fst.input_symbols().find(k.ilabel)]

    for key in targets.keys()[:100]:
        print "{} arcs from {} to {}, e.g: {}".format(len(targets[key]), state, key, ", ".join(targets[key][:5]))
    if len(targets.keys()) > 100:
        print "..."

def speech_to_text(acoustic_probs, ling_fst, alpha, beta, k):
    acoustic_fst = probs_to_fst(acoustic_probs, ling_fst.input_symbols(), 1./alpha)
    search_fst = pfst.compose(acoustic_fst, ling_fst)
    k_best_strings = get_words(pfst.shortestpath(search_fst, nshortest = k))
    def word_weighter(tup):
        return -len(tup[0].split())*beta + float(tup[1])

    return min(k_best_strings, key=word_weighter)[0]

def lev_dist(result, label):
    return edit_dist(result.split(), label.split())

def edit_dist(a, b):
    val_store = [[0 for j in range(len(b) + 1)] for i in range(len(a)+1)]
    edit_dist_(a, b, len(a), len(b), val_store)
    return val_store[len(a)][len(b)]

def edit_dist_(a, b, i, j, val_store):
    if i == 0:
        val_store[i][j] = j
    elif j == 0:
        val_store[i][j] = i
    else:
        edit_dist_(a, b, i-1, j-1, val_store)
        edit_dist_(a, b, i-1, j, val_store)
        edit_dist_(a, b, i, j-1, val_store)
        val_store[i][j] = min(val_store[i-1][j] + 1, val_store[i][j-1] + 1, val_store[i-1][j-1] + int(a[i-1]!=b[j-1]))

def probs_to_fst(probs, st, acoustic_weight):
    # create the fst with log semiring weights
    weight_type = "tropical"
    f = pfst.Fst()
    f.set_input_symbols(st)
    f.set_output_symbols(st)

    s = f.add_state()
    f.set_start(s)
    x_prev = s
    # iterate over the frame-level probs
    T = probs.shape[0]
    for i in range(T):
        x = f.add_state()
        frame_probs = probs[i]
        # add a sausage
        for phon in phons:
            # add each arc to the sausage
            phon_symb_idx = f.input_symbols().find(phon)
            phon_probs_idx = phon_symb_idx - 1
            weight = pfst.Weight(weight_type, -np.log(frame_probs[phon_probs_idx]) * acoustic_weight)
            f.add_arc(x_prev, pfst.Arc(phon_symb_idx, phon_symb_idx, weight, x))
        x_prev = x
    f.set_final(x)
    f.verify()
    return f




def create_frame_acceptor(phoneme_string, new_i_symbs):
    phon_list = phoneme_string.split()
    string_fst = pfst.Fst()
    string_fst.set_input_symbols(new_i_symbs)
    string_fst.set_output_symbols(new_i_symbs)
    start_st = string_fst.add_state()
    string_fst.set_start(start_st)
    prev_st = start_st
    for phon in phon_list:
        next_st = string_fst.add_state()
        string_fst.add_arc(prev_st, pfst.Arc(new_i_symbs.find(phon), new_i_symbs.find(phon), None, next_st))
        prev_st = next_st
    
    string_fst.set_final(next_st)
    return string_fst

def create_word_acceptor(word_list, st):
    
    sentence_fst = pfst.Fst()
    sentence_fst.set_input_symbols(st)
    sentence_fst.set_output_symbols(st)
    start_state = sentence_fst.add_state()
    sentence_fst.set_start(start_state)
    prev_state = start_state
    for word in word_list:
        next_state = sentence_fst.add_state()
        sentence_fst.add_arc(prev_state, pfst.Arc(st.find(word), st.find(word), None, next_state))
        prev_state = next_state
    
    sentence_fst.set_final(next_state)
    return sentence_fst

#assumes fst is acyclic
def get_words(n_short_fst):
    o_symbs = n_short_fst.output_symbols()
    start_st = n_short_fst.start()
    fst_zero = pfst.Weight.Zero(n_short_fst.weight_type())
    fst_one = pfst.Weight.One(n_short_fst.weight_type())
    #starts at start state with empty string, zero weight
    poss_paths = [("", start_st, fst_one)]
    final_paths = []

    #while there are still unfinished paths
    while poss_paths:
        new_partials = []
        for (part_word, st, tot_weight) in poss_paths:
            poss_arcs = [arc for arc in n_short_fst.arcs(st)]
            
            #if no outgoing arcs, st is a final state
            if not poss_arcs:
                final_paths.append((part_word, tot_weight))
            else:
                for arc in poss_arcs:
                    new_fragment = o_symbs.find(arc.olabel)
                    
                    if new_fragment != "<eps>":
                        part_word_new = part_word + new_fragment + " "
                        print new_fragment
                    else:
                        part_word_new = part_word

                    new_st = arc.nextstate
                    new_weight = pfst.times(tot_weight, arc.weight)

                    #adds new partial paths
                    new_partials.append((part_word_new, new_st, new_weight))

        poss_paths = new_partials
       
    return final_paths
def get_lang(lang_model_name):
    
#    def word_end_iter():
#        for i in range(40, 53):
#            yield (i, 0)
    
#    lex = pfst.Fst.read("../lex_model/lex-uw.fst")
#    i_table = lex.input_symbols()
#    o_table = lex.output_symbols()
    lang = pfst.Fst.read(lang_model_name + ".pru")
#    LG = pfst.compose(lex, lang)
#    LG = pfst.determinize(LG)
#    LG.minimize()
#    LG.rmepsilon()
    
#    for i in range(40,53):
#        LG.relabel_pairs(word_end_iter())
#        print "Relabelled <w{}>".format(i)
    
#    LG.set_input_symbols(st)

    return lang

def get_lex(max_words):
    D = nltk.corpus.cmudict.dict()
    S = map(lambda x: [x[0], float(x[1])], np.load("../unigram_scores.npz")["arr_0"][:max_words])
    print "First 10 items in S are:"
    print S[:10]
    print "Last 10 items in S are:"
    print S[-10:]
    
    def strip_num(phone):
        if phone[-1].isdigit():
            return phone[:-1]
        else:
            return phone
    
    word_to_phon_list = {} 
    for k, v in D.items():
        word_to_phon_list[k] = map(strip_num, v[0])
        
    phon_st = pfst.SymbolTable().read_text("phoneme-symb-table.txt")
    word_st = st = pfst.SymbolTable().read_text("../word-symb-table.txt")
    lex_fst = pfst.Fst()
    lex_fst.set_input_symbols(phon_st)
    lex_fst.set_output_symbols(word_st)
    s = lex_fst.add_state()
    lex_fst.set_start(s)
    lex_fst.set_final(s)

    for word, _ in S[:max_words]:
        phon_list = word_to_phon_list[word]
        current_state = s
        for i, phon in enumerate(phon_list):
            phon_symb_idx = lex_fst.input_symbols().find(phon)
            exists = False
            for arc in lex_fst.arcs(current_state):
                if phon_symb_idx == arc.ilabel:
                    current_state = arc.nextstate
                    exists = True
                    break
            if exists and i!=len(phon_list)-1:
                continue
            else:
                next_state = lex_fst.add_state()
                lex_fst.add_arc(current_state, pfst.Arc(phon_symb_idx, word_st.find("<eps>"), None, next_state))
            current_state = next_state

#        word_state = lex_fst.add_state()
        lex_fst.add_arc(current_state, pfst.Arc(0, word_st.find(word), None, s))
#        lex_fst.set_final(word_state)
#        lex_fst.closure()
#        lex_fst.write("reduced-lex.fst")
    return lex_fst
    
    
def get_rm_dupes():
    weight_type="tropical"
    rm_dupes = pfst.Fst()
    rm_dupes.set_input_symbols(pfst.SymbolTable().read_text("phoneme-symb-table-with-blank.txt"))
    rm_dupes.set_output_symbols(pfst.SymbolTable().read_text("phoneme-symb-table-with-blank.txt"))

    s = rm_dupes.add_state()
    rm_dupes.set_start(s)
    rm_dupes.set_final(s)
    weight_one = pfst.Weight.One(weight_type)

    for phon in phons:
        x = rm_dupes.add_state()
        rm_dupes.set_final(x)
        phon_symb_idx = rm_dupes.input_symbols().find(phon)
        rm_dupes.add_arc(s, pfst.Arc(phon_symb_idx, phon_symb_idx, weight_one, x))
        # Add arc for repeat phoneme
        rm_dupes.add_arc(x, pfst.Arc(phon_symb_idx, 0, weight_one, x))

    for state in rm_dupes.states():
        if state == 0:
            continue
        for next_state in rm_dupes.states():
            if next_state == 0 or next_state == state:
                continue
    #         next_phon_symb_idx = rm_dupes.input_symbols().find(phon)
            rm_dupes.add_arc(state, pfst.Arc(next_state, next_state, weight_one, next_state))
    return rm_dupes

def get_rm_blanks():
    weight_type="tropical"
    rm_blanks = pfst.Fst()
    rm_blanks.set_input_symbols(pfst.SymbolTable().read_text("phoneme-symb-table-with-blank.txt"))
    rm_blanks.set_output_symbols(pfst.SymbolTable().read_text("phoneme-symb-table.txt"))

    s = rm_blanks.add_state()
    rm_blanks.set_start(s)
    rm_blanks.set_final(s)
    weight_one = pfst.Weight.One(weight_type)

    rm_blanks.add_arc(s, pfst.Arc(rm_blanks.input_symbols().find("-"), 0, weight_one, s))
    for phon in phons[:-1]:
        #x = rm_blanks.add_state()
        phon_symb_idx = rm_blanks.input_symbols().find(phon)
        rm_blanks.add_arc(s, pfst.Arc(phon_symb_idx, phon_symb_idx, weight_one, s))
    return rm_blanks

def add_blank_loops(search_fst):
    i_table = search_fst.input_symbols()
    o_table = search_fst.output_symbols()
    new_blank_arcs = []
#    new_repeat_phon_arcs = []
    for state in search_fst.states():
        
        #adds the blank self-transitions to each state
        new_blank_arcs.append((state, pfst.Arc(i_table.find("-"), o_table.find("<eps>"), 0., state)))
        
        #adds the repeat phoneme self-transitions to each state
#        arc_iter = search_fst.arcs(state)
#        for arc in arc_iter:
#            arc_phon = arc.ilabel
#            if arc_phon != i_table.find("<eps>"):
#                new_repeat_phon_arcs.append((arc.nextstate, pfst.Arc(arc_phon, o_table.find("<eps>"), 0., arc.nextstate)))
                                            
    for pair in new_blank_arcs: 
        search_fst.add_arc(*pair)  
#    for pair in new_repeat_phon_arcs:
#        search_fst.add_arc(*pair)
    return search_fst


"""

def create_acceptor(phoneme_string):
    
    phon_list = phoneme_string.split()
    string_fst = pfst.Fst()
    string_fst.set_input_symbols(i_table)
    string_fst.set_output_symbols(i_table)
    start_st = string_fst.add_state()
    string_fst.set_start(start_st)
    prev_st = start_st
    for phon in phon_list:
        next_st = string_fst.add_state()
        string_fst.add_arc(prev_st, pfst.Arc(i_table.find(phon), i_table.find(phon), None, next_st))
        prev_st = next_st
    
    string_fst.set_final(next_st)
    return string_fst



"""