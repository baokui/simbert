from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
import sys
from sklearn.metrics import roc_auc_score
def getAcc(labels,preds,thr):
    y = [int(t>=thr) for t in preds]
    n = sum([y[i]==labels[i] for i in range(len(y))])
    return n/len(y)
# path_model='/search/odin/guobk/data/my_simbert_l4_sim-pretrain-mlmcse/latest_model.weights'
# path_data='/search/odin/guobk/data/Tab3_test/test-20210804-testdata.json'
# path_target='./log/eval-my_simbert_l4_sim-pretrain-mlmcse.log'
# config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config_re.json'
# dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab_re.txt'
# checkpoint_path=None
path_model,path_data,path_target,config_path,checkpoint_path,dict_path = sys.argv[1:]
if checkpoint_path=='None':
    checkpoint_path = None
model, seq2seq, encoder,tokenizer = get_model(config_path,checkpoint_path,dict_path)
model.load_weights(path_model)
with open(path_data,'r') as f:
    D = json.load(f)
def emb(encoder,Sents, batch_size = 128,length=128):
    V = []
    X, S = [], []
    for t in Sents:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X,length=length)
    S = sequence_padding(S,length=length)
    Z = encoder.predict([X, S],verbose=True)
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return Z
Sents = [d['content'] for d in D]
for d in D:
    Sents.extend(d['pos'])
    Sents.extend(d['neg'])
V_d = emb(encoder,Sents)
D_v = {Sents[i]:V_d[i] for i in range(len(Sents))}
labels = []
preds = []
for d in D:
    labels.extend([1]*len(d['pos']))
    labels.extend([0]*len(d['neg']))
    v0 = D_v[d['content']]
    v1 = [D_v[t] for t in d['pos']] + [D_v[t] for t in d['neg']]
    v1 = np.array(v1)
    s = v1.dot(v0)
    preds.extend(list(s))
nb_pos = sum(labels)
nb_neg = len(labels) - nb_pos
auc = roc_auc_score(labels,preds)
acc = getAcc(labels,preds,thr=0.5)
s = ['model: '+path_model]
s.append('num of pos and neg: %d, %d'%(nb_pos,nb_neg))
s.append('auc: %0.4f'%auc)
s.append('acc: %0.4f'%acc)
with open(path_target,'w') as f:
    f.write('\n'.join(s))