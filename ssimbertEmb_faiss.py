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
def getContent():
    '''
    SQL数据库
    host: mt.tugele.rds.sogou
    port: 3306
    user: tugele_new
    password: tUgele2017OOT

    表名字 ns_flx_wisdom_words_new
    :return:
    '''
    import pymysql
    conn = pymysql.connect(
        host='mt.tugele.rds.sogou',
        user='tugele_new',
        password='tUgele2017OOT',
        charset='utf8',
        port  = 3306,
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
        # ****python, 必须有一个游标对象， 用来给数据库发送sql语句， 并执行的.
        # 2. 创建游标对象，
    cur = conn.cursor()
    # 4). **************************数据库查询*****************************
    # sqli = 'SELECT * FROM tugele.ns_flx_wisdom_words_new'
    sqli = 'SELECT a.id,a.content,a.isDeleted,a.status FROM (tugele.ns_flx_wisdom_words_new a)'
    cur.execute('SET NAMES utf8mb4')
    cur.execute("SET CHARACTER SET utf8mb4")
    cur.execute("SET character_set_connection=utf8mb4")
    result = cur.execute(sqli)  # 默认不返回查询结果集， 返回数据记录数。
    info = cur.fetchall()  # 3). 获取所有的查询结果
    # print(info)
    # print(len(info))
    # 4. 关闭游标
    cur.close()
    # 5. 关闭连接
    conn.close()
    S = [[str(info[i][0]),info[i][1]] for i in range(len(info)) if info[i][2]==0 and info[i][3]==1]
    return S
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

path_model="/search/odin/guobk/data/my_simbert_l4/model_269.h5"
bert_model="chinese_simbert_L-4_H-312_A-12"
path_target = '/search/odin/guobk/data/faiss_search/simbert/'
if not os.path.exists(path_target):
    os.mkdir(path_target)
model, seq2seq, encoder,tokenizer = get_model(bert_model)
model.load_weights(path_model)
Docs = getContent()
SentsD = [d[1] for d in Docs]
SentsD,R = trim(SentsD)
Queries = random.sample(Docs,1000)
SentsQ = [d[1] for d in Queries]
def emb(encoder,Sents, batch_size = 128, maxlen = 64):
    V = []
    X, S = [], []
    for t in Sents:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X,length=maxlen)
    S = sequence_padding(S,length=maxlen)
    Z = encoder.predict([X, S],verbose=True)
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return Z

V_q = emb(encoder,SentsQ)
V_d = emb(encoder,SentsD)
R_d = []
for i in range(len(Docs)):
    v = ['%0.8f'%t for t in V_d[i]]
    v = [Docs[i][0]] + v
    R_d.append('\t'.join(v))
R_q = []
for i in range(len(Queries)):
    v = ['%0.8f'%t for t in V_q[i]]
    v = [Queries[i][0]] + v
    R_q.append('\t'.join(v))
with open(os.path.join(path_target,'Docs.txt'),'w') as f:
    f.write('\n'.join(R_d))
with open(os.path.join(path_target,'Queries.txt'),'w') as f:
    f.write('\n'.join(R_q))