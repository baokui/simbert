from simbert import *
import json
import numpy as np
model.load_weights(os.path.join(path_model,'model_069.h5'))
path_docs="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json"
# path_queries="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-test-0623.json"
path_queries = "/search/odin/guobk/data/bert_semantic/finetuneData_new_test/test-0623.json"
path_target="/search/odin/guobk/data/bert_semantic/finetuneData_new_test/test-0623-simbert.json"
with open(path_docs,'r') as f:
    D = json.load(f)
with open(path_queries,'r') as f:
    Q = json.load(f)
def emb(Sents, batch_size = 128):
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

SentsQ = [d['input'] for d in Q]
V_q = emb(SentsQ)
SentsD = [d['content'] for d in D]
V_d = emb(SentsD)
s = V_q[:1000].dot(np.transpose(V_d))
idx = np.argsort(-s,axis=-1)
maxRec = 10
for j in range(len(Queries)):
    score = [s[j][ii] for ii in idx[j][:maxRec]]
    contents = [SentsD[ii] for ii in idx[j][:maxRec]]
    Queries[j]['rec_simbert'] = [contents[k]+'\t%0.4f'%score[k] for k in range(len(score))]
with open(path_target,'w') as f:
    json.dump(Queries,f,ensure_ascii=False,indent=4)