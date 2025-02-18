from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
import sys
def write_excel(path_target,data,sheetname='Sheet1'):
    import xlwt
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet(sheetname)
    # 写入excel
    # 参数对应 行, 列, 值
    rows,cols = len(data),len(data[0])
    for i in range(rows):
        for j in range(cols):
            #worksheet.write(i, j, label=str(data[i][j]))
            worksheet.write(i, j, label=data[i][j])
    # 保存
    workbook.save(path_target)
# path_models = ['/search/odin/guobk/data/my_simbert/model_069.h5',\
#     '/search/odin/guobk/data/my_simbert_l4/model_269.h5',\
#         '/search/odin/guobk/data/my_simber_l6/latest_model.weights']
# tags = ['simbert_12-69','simbert_4','simbert_6']
# bert_models = ['chinese_L-12_H-768_A-12','chinese_simbert_L-4_H-312_A-12']
path_model,tag,path_docs,path_queries,maxQ,path_target,config_path,checkpoint_path,dict_path = sys.argv[1:]
if checkpoint_path=='None':
    checkpoint_path = None
model, seq2seq, encoder,tokenizer = get_model(config_path,checkpoint_path,dict_path)
model.load_weights(path_model)
with open(path_docs,'r') as f:
    D = json.load(f)
with open(path_queries,'r') as f:
    Q = json.load(f)
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
def simScore(v_q,V_d0):
    V_d = np.transpose(V_d0)
    batch_size = 100
    s = []
    idx = []
    i = 0
    while i<len(v_q):
        print(i,len(v_q))
        j = min(len(v_q),i+batch_size)
        v0 = v_q[i:j]
        s0 = v0.dot(V_d)
        idx0 = np.argsort(-s0,axis=-1)
        s.append(s0)
        idx.append(idx0)
        i = i + batch_size
    s = np.concatenate(s,axis=0)
    idx = np.concatenate(idx,axis=0)
    return s,idx


maxQ = int(maxQ)
Queries = Q[:maxQ]
maxRec = 10
SentsQ = [d['input'] for d in Queries]
SentsD = [d['content'] for d in D]
V_q = emb(encoder,SentsQ)
V_d = emb(encoder,SentsD)
# s = V_q.dot(np.transpose(V_d))
# idx = np.argsort(-s,axis=-1)
s,idx = simScore(V_q,V_d)

for j in range(len(Queries)):
    score = [s[j][ii] for ii in idx[j][:maxRec]]
    contents = [SentsD[ii] for ii in idx[j][:maxRec]]
    Queries[j][tag] = [contents[k]+'\t%0.4f'%score[k] for k in range(len(score))]
with open(path_target,'w') as f:
    json.dump(Queries,f,ensure_ascii=False,indent=4)

def test0():
    import json
    path_source = '/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
    path_source = '/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert-com.json'
    with open(path_source,'r') as f:
        D = json.load(f)
    R = [['index','query']]
    keys = [k for k in D[0] if 'rec_' in k]
    for k in keys:
        R[0].extend([k,k+'-score',k+'-acc'])
    # keys = ['rec_my_simbert_l4','rec_my_simbert_l4_sim0809','rec_my_simbert_l4_sim0809-flatnce','rec_my_simbert_l4_sim-pretrain-mlmcse']
    ii = 0
    maxRec = 5
    T = []
    for d in D:
        r00 = []
        t = {'input':d['input'],'rec':[]}
        for k in keys:
            r0 = d[k][:maxRec]
            r0 = r0 + ['']*(maxRec-len(r0))
            r00.append([t.split('\t')[:1]+['',''] for t in r0])
            t['rec'].extend([t.split('\t')[0] for t in r0])
        r = [[ii,d['input']]]
        for i in range(len(r00)):
            r[0].extend(r00[i][0])
        for j in range(1,maxRec):
            rr = ['','']
            for i in range(len(r00)):
                rr.extend(r00[i][j])
            r.append(rr)
        ii += 1
        R.extend(r)
        R.append(['' for _ in range(len(R[0]))])
        T.append(t)
    # #####################################
    # for i in range(len(R)):
    #     if R[i][2] and R[i][2][0]=='*':
    #         R[i][4] = 1
    #         R[i][2] = R[i][2][1:]
    #     if R[i][5] and R[i][5][0]=='*':
    #         R[i][7] = 1
    #         R[i][5] = R[i][5][1:]
    #####################################
    write_excel(path_source.replace('.json','.xls'),R)
    
    for t in T:
        t['rec'] = list(set(t['rec']))
    with open(path_source.replace('.json','-test.json'),'w') as f:
        json.dump(T,f,ensure_ascii=False,indent=4)
    
    with open(path_source.replace('.json','-test.json'),'r') as f:
        T = json.load(f)
    T0 = {}
    for t in T:
        rec = [s for s in t['rec'] if s]
        T0[t['input']] = [s[1:] for s in rec if s[0]=='0']
    R = [['index','query']]
    keys = [k for k in D[0] if 'rec_' in k]
    for k in keys:
        R[0].extend([k,k+'-score',k+'-acc'])
    ii = 0
    maxRec = 5
    for d in D:
        r00 = []
        t = {'input':d['input'],'rec':[]}
        for k in keys:
            r0 = d[k][:maxRec]
            r0 = r0 + ['']*(maxRec-len(r0))
            r00.append([t.split('\t')[:1]+['',''] for t in r0])
        for i in range(len(r00)):
            for j in range(len(r00[i])):
                if r00[i][j][0] in T0[d['input']]:
                    r00[i][j][1] = 1
                else:
                    r00[i][j][1] = 0
        r = [[ii,d['input']]]
        for i in range(len(r00)):
            r[0].extend(r00[i][0])
        for j in range(1,maxRec):
            rr = ['','']
            for i in range(len(r00)):
                rr.extend(r00[i][j])
            r.append(rr)
        ii += 1
        R.extend(r)
        R.append(['' for _ in range(len(R[0]))])
        R[-1][3] = 0
    write_excel(path_source.replace('.json','-test.xls'),R)
def test1():
    import json
    path_source = '/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-alpha.json'
    with open(path_source,'r') as f:
        D = json.load(f)
    R = read_excel('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert-com.xls',1)
    T = []
    i = 3
    while i < len(R):
        if R[i][1]=='':
            i+=1
            continue
        d = {'input':R[i][1]}
        r = {}
        for j in range(i,i+5):
            for k in range(2,len(R[0]),3):
                if R[j][k]!='':
                    r[R[j][k]] = str(int(R[j][k+1]))
        d['rec'] = r
        i = j+1
        T.append(d)
    keys = [k for k in D[0].keys() if 'alpha' in k]
    for  i in range(len(T)):
        r = []
        for k in keys:
            r.extend([t.split('\t')[0] for t in D[i][k]])
        r = list(set(r))
        for t in r:
            if t not in T[i]['rec']:
                T[i]['rec'][t] = ''
    with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert-com-test.json','w') as f:
        json.dump(T,f,ensure_ascii=False,indent=4)
    with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec-bert-com-test.json','r') as f:
        T1 = json.load(f)
    r = []
    for k in keys:
        r.extend([k,'score','acc'])
    R = [r]
    for i in range(len(T1)):
        r0 = []
        for j in range(5):
            r = []
            for k in keys:
                if j > len(D[i][k]):
                    r.extend(['','0',''])
                else:
                    s = D[i][k][j].split('\t')[0]
                    if s in T1[i]['rec']:
                        t = T1[i]['rec'][s]
                        if t=='':
                            t='0'
                    else:
                        t = '1'
                        print(s)
                    r.extend([s,int(t),''])
            r0.append(r)
        R.extend(r0)
        R.append(['' for _ in range(len(R[0]))])
    write_excel(path_source.replace('.json','-test2.xls'),R[:-1])