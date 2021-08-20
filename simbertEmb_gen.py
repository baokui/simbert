from model import get_model
import json
import numpy as np
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import AutoRegressiveDecoder
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
def gen_synonyms(text,synonyms_generator,tokenizer,encoder,seq2seq, n=100, k=20):
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

# path_source="/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json"
# path_target="/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-gen.json"
# tags="ori,alpha_0,alpha_0.1,alpha_0.25,alpha_0.5,alpha_0.75"
# path_models="/search/odin/guobk/data/my_simbert_l4/model_269.h5,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.1/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.25/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.5/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.75/latest_model.weights"
# config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
# checkpoint_path="None"
# dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
path_source,path_target,path_models,tags,config_path,checkpoint_path,dict_path = sys.argv[1:]
maxlen = 32
path_models = path_models.split(',')
tags = tags.split(',')
if checkpoint_path=='None':
    checkpoint_path = None
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

R = []
for s in SentsQ[len(R):]:
    d = {'input':s}
    for i in range(len(Models)):
        encoder = Models[i][2]
        seq2seq = Models[i][1]
        r = gen_synonyms(s, synonyms_generator,tokenizer,encoder,seq2seq, 10, 10)
        d[tags[i]] = r
    R.append(d)
with open(path_target,'w') as f:
    json.dump(R,f,ensure_ascii=False,indent=4)