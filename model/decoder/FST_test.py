import graphviz
import pydot
# import pywrapfst
import fst
import nltk
import re
import os

LexM = fst.read("lex_model/lex-uw.fst")
LM_expr = "^(.*)\.pru$"
folder_name = "lang_model"
file_list = [os.path.join(folder_name, fname) for fname in os.listdir(folder_name)]
pruned_models = [re.match(LM_expr, filename).group(1) for filename in file_list if re.match(LM_expr, filename)]

i_table = LexM.isyms
o_table = LexM.osyms

mod_name = "lang_model/3-gram-3"

LG = fst.read(mod_name + ".pi").copy()

test_word = fst.Acceptor(syms=i_table)
test_word.add_arc(0, 1, 'HH')
test_word.add_arc(1, 2, 'EY')
test_word[2].final = True

test_comp = test_word >> LG
