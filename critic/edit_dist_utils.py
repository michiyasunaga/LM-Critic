"""
Edit distance utils...

Originally by https://worksheets.codalab.org/worksheets/0x8fc01c7fc2b742fdb29c05669f0ad7d2
"""
from collections import defaultdict
import numpy as np
import random
import string
from itertools import permutations

def process_filetype(filetype):
    insert = (filetype // 1000) % 2 == 1
    delete = (filetype // 100) % 2 == 1
    substitute = (filetype // 10) % 2 == 1
    swap = filetype % 2 == 1
    return insert, delete, substitute, swap

def get_all_edit_dist_one(word, filetype = 1111, sub_restrict = None):
    """
    Allowable edit_dist_one perturbations:
        1. Insert any lowercase characer at any position other than the start
        2. Delete any character other than the first one
        3. Substitute any lowercase character for any other lowercase letter other than the start
        4. Swap adjacent characters
    We also include the original word. Filetype determines which of the allowable perturbations to use.
    """
    insert, delete, substitute, swap = process_filetype(filetype)
    #last_mod_pos is last thing you could insert before
    last_mod_pos = len(word) #- 1
    ed1 = set()
    if len(word) <= 2 or word[:1].isupper() or word[:1].isnumeric():
        return ed1
    for pos in range(1, last_mod_pos + 1): #can add letters at the end
        if delete and pos < last_mod_pos:
            deletion = word[:pos] + word[pos + 1:]
            ed1.add(deletion)
        if swap and pos < last_mod_pos - 1:
            #swapping thing at pos with thing at pos + 1
            swaped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            ed1.add(swaped)
        for letter in string.ascii_lowercase: #+"'-": #no need to add '-, as we want to corrupt good to bad
            if insert:
                #Insert right after pos - 1
                insertion = word[:pos] + letter + word[pos:]
                ed1.add(insertion)
            can_substitute = sub_restrict is None or letter in sub_restrict[word[pos]]
            if substitute and pos < last_mod_pos and can_substitute:
                substitution = word[:pos] + letter + word[pos + 1:]
                ed1.add(substitution)
    #Include original word
    # ed1.add(word)
    return ed1

def get_all_internal_permutations(word):
    if len(word) > 10:
        return set([word])
    first_char = word[0]
    last_char = word[-1]
    internal_chars = word[1:-1]
    internal_permutations = set()
    for int_perm in permutations(internal_chars):
        int_perm_str = ''.join(int_perm)
        perm = '{}{}{}'.format(first_char, int_perm_str, last_char)
        internal_permutations.add(perm)
    return internal_permutations

def sample_random_internal_permutations(word, n_perts = 5):
    #We try swapping everything with the second character...
    if len(word) < 4:
        return set([word])
    #iterate through positions between second and last
    perturbations = set()
    start = word[0]
    end = word[-1]
    middle = word[1:-1]
    for _ in range(n_perts):
        middle_list = list(middle)
        random.shuffle(middle_list)
        mixed_up_middle = ''.join(middle_list)
        perturbations.add('{}{}{}'.format(start, mixed_up_middle, end))
    return perturbations

def get_sorted_word(word):
    if len(word) < 3:
        sorted_word = word
    else:
        sorted_word = '{}{}{}'.format(word[0], ''.join(sorted(word[1:-1])), word[-1])
    return sorted_word

def get_sorted_word_set(word):
    if len(word) < 3:
        sorted_word = word
    else:
        sorted_word = '{}{}{}'.format(word[0], ''.join(sorted(word[1:-1])), word[-1])
    return set([sorted_word])


#Used to create agglomerative clusters.
def preprocess_ed1_neighbors(vocab, sub_restrict = None, filetype = 1111):
    vocab = set([word.lower() for word in vocab])
    typo2words = defaultdict(set)
    for word in vocab:
        ed1_typos = get_all_edit_dist_one(word, filetype = filetype, sub_restrict = sub_restrict)
        for typo in ed1_typos:
            typo2words[typo].add(word)

    word2neighbors = defaultdict(set)
    for typo in typo2words:
        for word in typo2words[typo]:
            word2neighbors[word] = word2neighbors[word].union(typo2words[typo])
    return word2neighbors

#Used to create agglomerative clusters.
def ed1_neighbors_mat(vocab, sub_restrict = None, filetype = 1111):
    vocab = [word.lower() for word in vocab]
    word2idx = dict([(word, i) for i, word in enumerate(vocab)])
    word2neighbors = preprocess_ed1_neighbors(vocab, sub_restrict = sub_restrict, filetype = filetype)
    edges = set()
    for word in word2neighbors:
        for neighbor in word2neighbors[word]:
            edge = [word, neighbor]
            edge.sort()
            edge = tuple(edge)
            edges.add(edge)
    edge_mat = np.zeros((len(vocab), len(vocab)), dtype = int)
    for edge in edges:
        vtx1, vtx2 = edge
        idx1, idx2 = word2idx[vtx1], word2idx[vtx2]
        edge_mat[idx1][idx2] = 1
        edge_mat[idx2][idx1] = 1
    return edge_mat



if __name__ == '__main__':
    while True:
        word = input("Enter a word: ")
        print("Total number of possible perturbations: {}".format(len(get_all_edit_dist_one(word))))
