"""
Originally by https://worksheets.codalab.org/worksheets/0x8fc01c7fc2b742fdb29c05669f0ad7d2
"""
import json
import os, sys
import re
import random
import numpy as np
from random import sample
from tqdm import tqdm
from collections import Counter

from critic.edit_dist_utils import get_all_edit_dist_one, sample_random_internal_permutations


try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = '.'
common_typo = json.load(open(f"{dir_path}/common_typo.json"))

random.seed(1234)
np.random.seed(1234)


class RandomPerturbationAttack(object):
    def __init__(self, attack_type = 'ed1'):
        self.cache = {} #{word: {0: set(), 1: set(),.. }, ..} #0=swap, 1=substitute, 2=delete, 3=insert
        self.n_types = 5
        self.attack_type = attack_type
    #
    def sample_perturbations(self, word, n_samples, types):
        if types is None:
            type_list = list(range(4)) * (n_samples//4) + list(np.random.choice(self.n_types, n_samples % self.n_types, replace=False))
        else:
            type_list = [sample(types,1)[0] for _ in range(n_samples)]
        type_count = Counter(type_list)
        perturbations = set()
        for type in type_count:
            if type not in self.cache[word]:
                continue
            if len(self.cache[word][type]) >= type_count[type]:
                perturbations.update(set(sample(self.cache[word][type], type_count[type])))
            else:
                perturbations.update(self.cache[word][type])
        return perturbations
    #
    def get_perturbations(self, word, n_samples, types=None):
        if word not in self.cache:
            self.cache[word] = {}
            if word[0].islower():
                for type in range(4):
                    self.cache[word][type] = get_all_edit_dist_one(word, 10**type)
                if word in common_typo:
                    self.cache[word][4] = set(common_typo[word])
            elif word[0].isupper():
                if word in common_typo:
                    self.cache[word][4] = set(common_typo[word])
        if self.attack_type == 'ed1':
            perturbations = self.sample_perturbations(word, n_samples, types)
        else:
            raise NotImplementedError("Attack type: {} not implemented yet".format(self.attack_type))
        return perturbations
    #
    def name(self):
        return 'RandomPerturbationAttack'


word_attack = RandomPerturbationAttack()


def _tokenize(sent):
    toks = []
    word_idxs = []
    for idx, match in enumerate(re.finditer(r'([a-zA-Z]+)|([0-9]+)|.', sent)):
        tok = match.group(0)
        toks.append(tok)
        if len(tok) > 2 and tok.isalpha() and (tok[0].islower()):
            word_idxs.append(idx)
    return toks, word_idxs

def _detokenize(toks):
    return ''.join(toks)

def get_local_neighbors_char_level(sent, max_n_samples=500):
    words, word_idxs = _tokenize(sent)
    n_samples = min(len(word_idxs) *20, max_n_samples)
    sent_perturbations = set()
    if len(word_idxs) == 0:
        return sent_perturbations
    for _ in range(500):
        word_idx = sample(word_idxs, 1)[0]
        words_cp = words[:]
        word_perturbations = list(word_attack.get_perturbations(words_cp[word_idx], n_samples=1))
        if len(word_perturbations) > 0:
            words_cp[word_idx] = word_perturbations[0]
            sent_perturbed = _detokenize(words_cp)
            if sent_perturbed != sent:
                sent_perturbations.add(sent_perturbed)
        if len(sent_perturbations) == n_samples:
            break
    #Adding common typos such as 's'
    for word_idx in word_idxs:
        words_cp = words[:]
        word = words_cp[word_idx]
        if len(word) > 2 and word[0].islower():
            words_cp[word_idx] = word +'s'
            sent_perturbed = _detokenize(words_cp)
            if sent_perturbed != sent:
                sent_perturbations.add(sent_perturbed)
            words_cp[word_idx] = word[:-1]
            sent_perturbed = _detokenize(words_cp)
            if sent_perturbed != sent:
                sent_perturbations.add(sent_perturbed)
    if len(sent_perturbations) > max_n_samples:
        sent_perturbations = list(sent_perturbations)
        np.random.shuffle(sent_perturbations)
        sent_perturbations = set(sent_perturbations[:max_n_samples])
    return sent_perturbations



from critic.PIE.word_level_perturb import WordLevelPerturber_all, WordLevelPerturber_refine
from utils.text_utils import detokenize_sent

def get_local_neighbors_word_level(sent_toked, max_n_samples=500, mode='refine'):
    """ sent_toked is tokenized by spacy """
    n_samples = min(len(sent_toked) *20, max_n_samples)
    orig_sent = ' '.join(sent_toked)
    orig_sent_detok = detokenize_sent(orig_sent)
    if mode == 'refine':
        ptb = WordLevelPerturber_refine(orig_sent)
    else:
        ptb = WordLevelPerturber_all(orig_sent)
    sent_perturbations = set()
    for _ in range(500):
        sent_perturbed = ptb.perturb()
        if sent_perturbed != orig_sent:
            sent_perturbed_detok = detokenize_sent(sent_perturbed)
            sent_perturbations.add(sent_perturbed_detok)
        if len(sent_perturbations) == n_samples:
            break
    assert len(sent_perturbations) <= max_n_samples
    return sent_perturbations, orig_sent_detok
