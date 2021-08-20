from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
import sys
import random
import os
class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    # @AutoRegressiveDecoder.set_rtype('probas')
    #def predict(self, inputs, output_ids, step):
    def predict(self, inputs, output_ids, states=0, temperature=1, probas='p'):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1],states
    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]
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
def gen_synonyms(text,synonyms_generator,tokenizer,encoder, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    效果：
        >>> gen_synonyms(u'微信和支付宝哪个好？')
        [
            u'微信和支付宝，哪个好?',
            u'微信和支付宝哪个好',
            u'支付宝和微信哪个好',
            u'支付宝和微信哪个好啊',
            u'微信和支付宝那个好用？',
            u'微信和支付宝哪个好用',
            u'支付宝和微信那个更好',
            u'支付宝和微信哪个好用',
            u'微信和支付宝用起来哪个好？',
            u'微信和支付宝选哪个好',
        ]
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


path_source,path_target,path_models,tags,config_path,checkpoint_path,dict_path = sys.argv[1:]
path_models = path_models.split(',')
tags = tags.split(',')
if checkpoint_path=='None':
    checkpoint_path = None
if not os.path.exists(path_target):
    os.mkdir(path_target)
Models = [get_model(config_path,checkpoint_path,dict_path) for i in range(len(path_models))]
tokenizer = Models[0][3]
synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)
for i in range(len(path_models)):
    Models[i][0].load_weights(path_models[i])
with open(path_source,'r') as f:
    Q = json.load(f)
SentsQ = [d['input'] for d in Q[:100]]

for s in SentsQ:
    d = {'input':s}
    for i in range(len(Models)):
        encoder = Models[i][2]
        r = gen_synonyms(s, synonyms_generator,tokenizer,encoder, 10, 10)
        d[tags[i]] = r