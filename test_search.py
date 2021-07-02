from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
import sys
# path_models = ['/search/odin/guobk/data/my_simbert/model_069.h5',\
#     '/search/odin/guobk/data/my_simbert_l4/model_269.h5',\
#         '/search/odin/guobk/data/my_simber_l6/latest_model.weights']
# tags = ['simbert_12-69','simbert_4','simbert_6']
# bert_models = ['chinese_L-12_H-768_A-12','chinese_simbert_L-4_H-312_A-12']
path_model,bert_model,tag,path_docs,path_queries,maxQ = sys.argv[1:]
model, seq2seq, encoder,tokenizer = get_model(bert_model)
model.load_weights(path_model)
with open(path_docs,'r') as f:
    D = json.load(f)
with open(path_queries,'r') as f:
    Q = json.load(f)
def emb(encoder,Sents, batch_size = 128):
    V = []
    i0 = 0
    i = 0
    i0 = i*batch_size
    i1 = (i+1)*batch_size
    while i0<len(Sents):
        r = Sents[i0:i1]
        X, S = [], []
        for t in r:
            x, s = tokenizer.encode(t)
            X.append(x)
            S.append(s)
        X = sequence_padding(X)
        S = sequence_padding(S)
        Z = encoder.predict([X, S])
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        if i==0:
            V = Z
        else:
            V = np.concatenate((V,Z),axis=0)
        i += 1
        i0 = i*batch_size
        i1 = (i+1)*batch_size
    return V
maxRec = int(maxQ)
Queries = Q[:maxRec]
SentsQ = [d['input'] for d in Queries]
SentsD = [d['content'] for d in D]
V_q = emb(encoder,SentsQ)
V_d = emb(encoder,SentsD)
s = V_q.dot(np.transpose(V_d))
idx = np.argsort(-s,axis=-1)
for j in range(len(Queries)):
    score = [s[j][ii] for ii in idx[j][:maxRec]]
    contents = [SentsD[ii] for ii in idx[j][:maxRec]]
    Queries[j][tag] = [contents[k]+'\t%0.4f'%score[k] for k in range(len(score))]
with open(path_target,'w') as f:
    json.dump(Queries,f,ensure_ascii=False,indent=4)
