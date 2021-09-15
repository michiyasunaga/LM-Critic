import re
import os
import sys
import json
import numpy as np
import editdistance
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, '.')
from utils.text_utils import detokenize_sent
from critic.critic import run_gpt2, gpt2_critic

def load_data():
    data_path = 'eval_critic/eval_data.jsonl'
    good_sents, bad_sents = [], []
    for line in open(data_path):
        obj = json.loads(line)
        good_sents.append(obj['good'])
        bad_sents.append(obj['bad'])
    return good_sents, bad_sents

good_sents, bad_sents = load_data()



def get_logps(sents):
    final = []
    for start in tqdm(range(0, len(sents), 100)):
        sents_sub = sents[start: start+100]
        sents_sub_detok = [detokenize_sent(sent) for sent in sents_sub]
        logps = run_gpt2(sents_sub_detok)
        assert logps is not None
        for i in range(len(sents_sub)):
            final.append({'sent': sents_sub[i], 'sent_detok': sents_sub_detok[i], 'logp': float(logps[i])})
    return final

def evaluate_logp():
    """
    Check whether log p(bad_sent) < log p(good_sent)
    """
    good_logps = get_logps(good_sents)
    bad_logps  = get_logps(bad_sents)
    accs = []
    for good, bad in zip(good_logps, bad_logps):
        accs.append(int(bad['logp'] < good['logp']))
    avg_acc = float(sum(accs))/len(accs)
    print (f'log p(bad) < log p(good)? {sum(accs)} / {len(accs)} = {avg_acc:.3f}')
    return good_logps, bad_logps

good_logps, bad_logps = evaluate_logp()
# log p(bad) < log p(good)? 555 / 586 = 0.947


def compute_metrics(good_accs, bad_accs):
    goodP = float(sum(good_accs))/(len(bad_accs)-sum(bad_accs)+sum(good_accs))
    goodR = float(sum(good_accs))/len(good_accs)
    goodF05 = (1+0.5**2) * float(goodP * goodR)/((0.5**2 * goodP) + goodR)
    badP = float(sum(bad_accs))/(len(good_accs)-sum(good_accs)+sum(bad_accs))
    badR = float(sum(bad_accs))/len(bad_accs)
    badF05 = (1+0.5**2) * float(badP * badR)/((0.5**2 * badP) + badR)
    print (f'  Good precision = {sum(good_accs)} / {(len(bad_accs)-sum(bad_accs)+sum(good_accs))} = {goodP:.3f}')
    print (f'  Good recall    = {sum(good_accs)} / {len(good_accs)} = {goodR:.3f}')
    print (f'  Good F0.5      = {goodF05:.3f}')
    print (f'  Bad precision  = {sum(bad_accs)} / {(len(good_accs)-sum(good_accs)+sum(bad_accs))} = {badP:.3f}')
    print (f'  Bad recall     = {sum(bad_accs)} / {len(bad_accs)} = {badR:.3f}')
    print (f'  Bad F0.5       = {badF05:.3f}')
    return {'goodP': goodP, 'goodR': goodR, 'goodF05': goodF05, 'badP': badP, 'badR': badR, 'badF05': badF05}

def evaluate_baseline_critic():
    threshold = np.mean([elm['logp'] for elm in good_logps + bad_logps])
    good_accs, bad_accs = [], []
    for obj in good_logps:
        pred = int(obj['logp'] > threshold)
        good_accs.append(pred==1)
    for obj in bad_logps:
        pred = int(obj['logp'] > threshold)
        bad_accs.append(pred==0)
    print ('\nBaseline critic:')
    stats = compute_metrics(good_accs, bad_accs)
    json.dump(stats, open('baseline_critic.stats.json', 'w'), indent=2)

evaluate_baseline_critic()
# Baseline critic:
#   Good precision = 365 / 668 = 0.546
#   Good recall    = 365 / 586 = 0.623
#   Good F0.5      = 0.560
#   Bad precision  = 283 / 504 = 0.562
#   Bad recall     = 283 / 586 = 0.483
#   Bad F0.5       = 0.544


def evaluate_LM_Critic():
    good_accs, bad_accs = [], []
    for obj in tqdm(good_logps):
        res = gpt2_critic(obj['sent_detok'], verbose=0, seed=1, n_samples=100, word_level_mode='refine')
        pred = int(res[0])
        good_accs.append(pred==1)
    for obj in tqdm(bad_logps):
        res = gpt2_critic(obj['sent_detok'], verbose=0, seed=1, n_samples=100, word_level_mode='refine')
        pred = int(res[0])
        bad_accs.append(pred==0)
    print ('\nLM-Critic:')
    stats = compute_metrics(good_accs, bad_accs)
    json.dump(stats, open('lm_critic.stats.json', 'w'), indent=2)

evaluate_LM_Critic()
# LM-Critic: (there is variance due to the randomness of sampling, some variation in GPT2 return score)
#   Good precision = 446 / 654 = 0.682
#   Good recall    = 446 / 586 = 0.761
#   Good F0.5      = 0.696
#   Bad precision  = 378 / 518 = 0.730
#   Bad recall     = 378 / 586 = 0.645
#   Bad F0.5       = 0.711
