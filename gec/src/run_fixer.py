import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

sys.path.insert(0, '..')
from utils.text_utils import detokenize_sent
from utils.spacy_tokenizer import spacy_tokenize_gec, spacy_tokenize_bea19

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path')
parser.add_argument('-i', '--input_path')
parser.add_argument('-o', '--output_path')
parser.add_argument('--bea19', action='store_true')
args = parser.parse_args()


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(args.model_path, force_bos_token_to_be_generated=True)
model.eval()
model.cuda()


def run_model(sents):
    num_ret_seqs = 10
    inp_max_len = 66
    batch = [tokenizer(s, return_tensors='pt', padding='max_length', max_length=inp_max_len) for s in sents]
    oidx2bidx = {} #original index to final batch index
    final_batch = []
    for oidx, elm in enumerate(batch):
        if elm['input_ids'].size(1) <= inp_max_len:
            oidx2bidx[oidx] = len(final_batch)
            final_batch.append(elm)
    batch = {key: torch.cat([elm[key] for elm in final_batch], dim=0) for key in final_batch[0]}
    with torch.no_grad():
        generated_ids = model.generate(batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                                num_beams=10, num_return_sequences=num_ret_seqs, max_length=65)
    _out = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)
    outs = []
    for i in range(0, len(_out), num_ret_seqs):
        outs.append(_out[i:i+num_ret_seqs])
    final_outs = [[sents[oidx]] if oidx not in oidx2bidx else outs[oidx2bidx[oidx]] for oidx in range(len(sents))]
    return final_outs


def run_for_wiki_yahoo_conll():
    sents = [detokenize_sent(l.strip()) for l in open(args.input_path)]
    b_size = 40
    outs = []
    for j in tqdm(range(0, len(sents), b_size)):
        sents_batch = sents[j:j+b_size]
        outs_batch = run_model(sents_batch)
        for sent, preds in zip(sents_batch, outs_batch):
            preds_detoked = [detokenize_sent(pred) for pred in preds]
            preds = [' '.join(spacy_tokenize_gec(pred)) for pred in preds_detoked]
            outs.append({'src': sent, 'preds': preds})
    os.system('mkdir -p {}'.format(os.path.dirname(args.output_path)))
    with open(args.output_path, 'w') as outf:
        for out in outs:
            print (out['preds'][0], file=outf)


def run_for_bea19():
    sents = [detokenize_sent(l.strip()) for l in open(args.input_path)]
    b_size = 40
    outs = []
    for j in tqdm(range(0, len(sents), b_size)):
        sents_batch = sents[j:j+b_size]
        outs_batch = run_model(sents_batch)
        for sent, preds in zip(sents_batch, outs_batch):
            preds_detoked = [detokenize_sent(pred) for pred in preds]
            preds = [' '.join(spacy_tokenize_bea19(pred)) for pred in preds_detoked]
            outs.append({'src': sent, 'preds': preds})
    os.system('mkdir -p {}'.format(os.path.dirname(args.output_path)))
    with open(args.output_path, 'w') as outf:
        for out in outs:
            print (out['preds'][0], file=outf)


if args.bea19:
    run_for_bea19()
else:
    run_for_wiki_yahoo_conll()
