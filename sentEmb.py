from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
import sys
import random
import os
def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False
def trim(S0):
    S = []
    R = []
    for s in S0:
        t = sum([_is_chinese_char(tt) for tt in s[:8]])
        if t==0:
            idx = 8
            while idx < len(s) and not _is_chinese_char(s[idx]):
                idx += 1
            S.append(s[idx:])
            R.append([s,s[idx:]])
        else:
            S.append(s)
    return S,R
class simbert_emb(object):
    def __init__(self):
        path_model="/search/odin/guobk/data/my_simbert_l4/model_269.h5"
        bert_model="chinese_simbert_L-4_H-312_A-12"
        self.model, seq2seq, self.encoder,self.tokenizer = get_model(bert_model)
        self.model.load_weights(path_model)
    def emb(self,Sents, maxlen = 64):
        Sents,_ = trim(Sents)
        V = []
        X, S = [], []
        for t in Sents:
            x, s = tokenizer.encode(t)
            X.append(x)
            S.append(s)
        X = sequence_padding(X,length=maxlen)
        S = sequence_padding(S,length=maxlen)
        Z = self.encoder.predict([X, S],verbose=True)
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        return Z
