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
    with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json','r') as f:
        D = json.load(f)
    R = [['index','query','base','score','accuracy','base-dataEn','score','accuracy','flatnce','score','accuracy','pretrain','score','accuracy']]
    keys = ['rec_my_simbert_l4','rec_my_simbert_l4_sim0809','rec_my_simbert_l4_sim0809-flatnce','rec_my_simbert_l4_sim-pretrain-mlmcse']
    ii = 0
    maxRec = 5
    for d in D:
        r00 = []
        for k in keys:
            r0 = d[k][:maxRec]
            r0 = r0 + ['']*(maxRec-len(r0))
            r00.append([t.split('\t')+[''] for t in r0])
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
    # #####################################
    # for i in range(len(R)):
    #     if R[i][2] and R[i][2][0]=='*':
    #         R[i][4] = 1
    #         R[i][2] = R[i][2][1:]
    #     if R[i][5] and R[i][5][0]=='*':
    #         R[i][7] = 1
    #         R[i][5] = R[i][5][1:]
    #####################################
    write_excel('/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json.xls',R)
                