import json
import random
path_target = '/search/odin/guobk/data/simcse/20210621/train_simbert.json'
path_data = '/search/odin/guobk/data/simcse/20210621/train.txt'
with open(path_data,'r') as f:
    S = f.read().strip().split('\n')
S = [s.split('\t') for s in S]
D = {}
for s in S:
    if s[0] in D:
        D[s[0]].append(s[1])
    else:
        D[s[0]] = [s[1]]
R = []
for d in D:
    R.append({"text":d,"synonyms":D[d]})
T = [json.dumps(r,ensure_ascii=False) for r in R]
random.shuffle(T)
with open(path_target,'w') as f:
    f.write('\n'.join(T))