"""
Word-level perturbation generator.

Originally by https://github.com/awasthiabhijeet/PIE/tree/master/errorify
"""
import os
import math
import pickle
import random
import editdistance
from numpy.random import choice as npchoice
from collections import defaultdict


try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = '.'

VERBS = pickle.load(open(f'{dir_path}/verbs.p', 'rb'))
COMMON_INSERTS = set(pickle.load(open(f'{dir_path}/common_inserts.p', 'rb'))) #common inserts *to fix a sent*
COMMON_DELETES = pickle.load(open(f'{dir_path}/common_deletes.p','rb')) #common deletes *to fix a sent*
_COMMON_REPLACES = pickle.load(open(f'{dir_path}/common_replaces.p', 'rb')) #common replacements *to errorify a sent*



COMMON_REPLACES = {}
for src in _COMMON_REPLACES:
    for tgt in _COMMON_REPLACES[src]:
        if (src=="'re" and tgt=="are") or (tgt=="'re" and src=="are"):
            continue
        ED = editdistance.eval(tgt, src)
        if ED > 2:
            continue
        longer = max(len(src), len(tgt))
        if float(ED)/longer >= 0.5:
            continue
        if tgt not in COMMON_REPLACES:
            COMMON_REPLACES[tgt] = {}
        COMMON_REPLACES[tgt][src] = _COMMON_REPLACES[src][tgt]


VERBS_refine = defaultdict(list)
for src in VERBS:
    for tgt in VERBS[src]:
        ED = editdistance.eval(tgt, src)
        if ED > 2:
            continue
        longer = max(len(src), len(tgt))
        if float(ED)/longer >= 0.5:
            continue
        VERBS_refine[src].append(tgt)



class WordLevelPerturber_all:
    def __init__(self, sentence: str):
        self.original_sentence = sentence.rstrip()
        self.sentence = self.original_sentence
        self.tokenized = None
        self.tokenize()

    def tokenize(self):
        self.tokenized = self.sentence.split()

    def orig(self):
        return self.original_sentence

    def _insert(self):
        """Insert a commonly deleted word."""
        if len(self.tokenized) > 0:
            insertable = list(range(len(self.tokenized)))
            index = random.choice(insertable)
            plist = list(COMMON_DELETES.values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]
            # Choose a word
            ins_word = npchoice(list(COMMON_DELETES.keys()), p=plist)
            self.tokenized.insert(index,ins_word)
        return ' '.join(self.tokenized)

    def _mod_verb(self, redir=True):
        if len(self.tokenized) > 0:
            verbs = [i for i, w in enumerate(self.tokenized) if w in VERBS]
            if not verbs:
                if redir:
                    return self._replace(redir=False)
                return self.sentence
            index = random.choice(verbs)
            word = self.tokenized[index]
            if not VERBS[word]:
                return self.sentence
            repl = random.choice(VERBS[word])
            self.tokenized[index] = repl
        return ' '.join(self.tokenized)

    def _delete(self):
        """Delete a commonly inserted word."""
        if len(self.tokenized) > 1:
            toks_len = len(self.tokenized)
            toks = self.tokenized
            deletable = [i for i, w in enumerate(toks) if w in COMMON_INSERTS]
            if not deletable:
                return self.sentence
            index = random.choice(deletable)
            del self.tokenized[index]
        return ' '.join(self.tokenized)

    def _replace(self, redir=True):
        if len(self.tokenized) > 0:
            deletable = [i for i, w in enumerate(self.tokenized) if (w in COMMON_REPLACES)]
            if not deletable:
                if redir:
                    return self._mod_verb(redir=False)
                return self.sentence
            index = random.choice(deletable)
            word = self.tokenized[index]
            if not COMMON_REPLACES[word]:
                return self.sentence
            # Normalize probabilities
            plist = list(COMMON_REPLACES[word].values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]
            # Choose a word
            repl = npchoice(list(COMMON_REPLACES[word].keys()), p=plist)
            self.tokenized[index] = repl
        return ' '.join(self.tokenized)

    def perturb(self):
        count = 1
        orig_sent = self.sentence
        for x in range(count):
            perturb_probs = [.30,.30,.30,.10]
            perturb_fun = npchoice([self._insert, self._mod_verb, self._replace, self._delete],p=perturb_probs)
            self.sentence = perturb_fun()
            self.tokenize()
        res_sentence = self.sentence
        self.sentence = self.original_sentence
        self.tokenize()
        return res_sentence


class WordLevelPerturber_refine:
    def __init__(self, sentence: str):
        self.original_sentence = sentence.rstrip()
        self.sentence = self.original_sentence
        self.tokenized = None
        self.tokenize()

    def tokenize(self):
        self.tokenized = self.sentence.split()

    def orig(self):
        return self.original_sentence

    def _insert(self):
        """Insert a commonly deleted word."""
        if len(self.tokenized) > 0:
            insertable = list(range(len(self.tokenized)))
            index = random.choice(insertable)
            plist = list(COMMON_DELETES.values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]
            # Choose a word
            ins_word = npchoice(list(COMMON_DELETES.keys()), p=plist)
            self.tokenized.insert(index,ins_word)
        return ' '.join(self.tokenized)

    def _mod_verb(self, redir=True):
        if len(self.tokenized) > 0:
            verbs = [i for i, w in enumerate(self.tokenized) if w in VERBS_refine]
            if not verbs:
                if redir:
                    return self._replace(redir=False)
                return self.sentence
            index = random.choice(verbs)
            word = self.tokenized[index]
            if not VERBS_refine[word]:
                return self.sentence
            repl = random.choice(VERBS_refine[word])
            self.tokenized[index] = repl

        return ' '.join(self.tokenized)

    def _delete(self):
        """Delete a commonly inserted word."""
        if len(self.tokenized) > 1:
            toks_len = len(self.tokenized)
            toks = self.tokenized
            deletable = [i for i, w in enumerate(toks) if (w in COMMON_INSERTS) and (i>0 and toks[i-1].lower() == toks[i].lower())]
            if not deletable:
                return self.sentence
            index = random.choice(deletable)
            del self.tokenized[index]
        return ' '.join(self.tokenized)

    def _replace(self, redir=True):
        def _keep(i,w):
            if w.lower() in {"not", "n't"}:
                return True
            return False

        if len(self.tokenized) > 0:
            deletable = [i for i, w in enumerate(self.tokenized) if (w in COMMON_REPLACES) and (not _keep(i,w))]
            if not deletable:
                if redir:
                    return self._mod_verb(redir=False)
                return self.sentence

            index = random.choice(deletable)
            word = self.tokenized[index]
            if not COMMON_REPLACES[word]:
                return self.sentence

            # Normalize probabilities
            plist = list(COMMON_REPLACES[word].values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]

            # Choose a word
            repl = npchoice(list(COMMON_REPLACES[word].keys()), p=plist)
            self.tokenized[index] = repl

        return ' '.join(self.tokenized)

    def perturb(self):
        count = 1
        orig_sent = self.sentence
        for x in range(count):
            perturb_probs = [.30,.30,.30,.10]
            perturb_fun = npchoice([self._insert, self._mod_verb, self._replace, self._delete],p=perturb_probs)
            self.sentence = perturb_fun()
            self.tokenize()
        res_sentence = self.sentence
        self.sentence = self.original_sentence
        self.tokenize()
        return res_sentence
