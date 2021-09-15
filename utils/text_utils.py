import re
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

def handle_dounble_quote(sent):
    cur_str = ''
    exp_left = True
    ignore_space = False
    for char in sent:
        if char == '"':
            if exp_left: #this is a left "
                cur_str = cur_str.rstrip() + ' "'
                exp_left = (not exp_left)
                ignore_space = True
            else: #this is a right "
                cur_str = cur_str.rstrip() + '" '
                exp_left = (not exp_left)
                ignore_space = False
        else:
            if ignore_space: #expecting right
                if char == ' ':
                    continue
                else:
                    cur_str = cur_str + char
                    ignore_space = False
            else:
                cur_str = cur_str + char
    cur_str = cur_str.strip()
    cur_str = re.sub(r'[ ]+', ' ', cur_str)
    return cur_str

def postprocess_space(sent):
    sent = re.sub(r'[ ]+\.', '.', sent)
    sent = re.sub(r'[ ]+,', ',', sent)
    sent = re.sub(r'[ ]+!', '!', sent)
    sent = re.sub(r'[ ]+\?', '?', sent)
    sent = re.sub(r'\([ ]+', '(', sent)
    sent = re.sub(r'[ ]+\)', ')', sent)
    sent = re.sub(r' \'s( |\.|,|!|\?)', r"'s\1", sent)
    sent = re.sub(r'n \'t( |\.|,|!|\?)', r"n't\1", sent)
    return sent

def detokenize_sent(sent):
    #Clean raw sent
    sent = re.sub(r'\' s ', '\'s ', sent)
    toks = sent.split()
    if len([1 for t in toks if t=="'"]) % 2 == 0:
        toks = ['"' if t=="'" else t for t in toks]
    sent = ' '.join(toks)
    #
    sents = sent_tokenize(sent)
    final_sents = []
    for _sent in sents:
        _sent = detokenizer.detokenize(_sent.split())
        res = handle_dounble_quote(_sent)
        if res == -1:
            print ('unbalanced double quote')
            print (_sent)
        else:
            _sent = res
        final_sents.append(_sent)
    sent = ' '.join(final_sents)
    sent = postprocess_space(sent)
    return sent
