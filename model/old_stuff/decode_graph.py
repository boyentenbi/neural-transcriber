import pynini
import pywrapfst as fst

def get_paths(decode_graph, isymbs, osymbs, phoneme_list):
    phoneme_fst = pynini.acceptor(" ".join(phoneme_list), token_type = isymbs)
    return [path for path in pynini.compose(phoneme_fst, decode_graph).paths(input_token_type=isymbs, output_token_type=osymbs)]

def get_all_paths(decode_graph, best_phonemes):
    isymbs = decode_graph.input_symbols()
    osymbs = decode_graph.output_symbols()
    all_paths = []
    for phon_list in best_phonemes:
        all_paths.extend(get_paths(decode_graph, isymbs, osymbs, phon_list))

    return all_paths

def get_best_paths(decoder_paths, k):
    sorted_paths = sorted(decoder_paths, key=lambda x : float(x[2]))
    return sorted_paths[:k]
