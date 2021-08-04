import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
tf.reset_default_graph()
bert_model = 'chinese_simbert_L-4_H-312_A-12'
checkpoint_path = '/search/odin/guobk/data/model/{}/bert_model.ckpt'.format(bert_model)
checkpoint_path='/search/odin/guobk/data/model/pretrainCom/ckpt/model.ckpt-595914'
model_reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()
vars = [k for k in var_dict]
vars = sorted(vars)
D_map = {}
for k in vars:
    if 'adam' in k:
        continue
    D_map[k+':0'] = model_reader.get_tensor(k)

import json
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout
import os
import random
import sys
tf.reset_default_graph()
# bert配置
bert_model = 'chinese_simbert_L-4_H-312_A-12'
config_path = '/search/odin/guobk/data/model/{}/bert_config.json'.format(bert_model)
# checkpoint_path = '/search/odin/guobk/data/model/{}/bert_model.ckpt'.format(bert_model)
checkpoint_path = None
dict_path = '/search/odin/guobk/data/model/{}/vocab.txt'.format(bert_model)
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False,
)
t_vars0 = tf.trainable_variables()
t_vars = [t.name for t in t_vars0]

vars0 = [t for t in D_map]
D_vars0 = {}
for t in vars0:
    if 'dense' in t:
        if 'dense_' not in t:
            D_vars0[t.replace('dense','dense_1')] = t
        else:
            idx0 = t.index('dense_')
            idx1 = idx0+t[idx0:].index('/')
            idx = int(t[idx0+6:idx1])
            D_vars0[t.replace('dense_'+str(idx),'dense_'+str(idx+1))] = t
    else:
        D_vars0[t] = t
Map = {}
vars1 = []
for t in t_vars:
    if t in D_vars0:
        Map[t] = D_vars0[t]
        vars0.remove(D_vars0[t])
    else:
        # if 'FeedForward/dense' in t:
        #     s = t[:31]
        #     for tt in vars0:
        #         if tt[:31]==s and (('kernel' in t and 'kernel' in tt) or ('bias' in t and 'bias' in tt)):
        #             Map[t] = tt
        #             vars0.remove(tt)
        #             break
        # elif 'MultiHeadSelfAttention/dense' in t:
        #     s = t[:42]
        #     for tt in vars0:
        #         if tt[:42]==s and (('kernel' in t and 'kernel' in tt) or ('bias' in t and 'bias' in tt)):
        #             Map[t]=tt
        #             vars0.remove(tt)
        #             break
        # else:
        vars1.append(t)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
v_tf = t_vars0
for i in range(len(v_tf)):
    if v_tf[i].name in Map and 'MLM' not in v_tf[i].name:
        w = D_map[Map[v_tf[i].name]]
        if v_tf[i].name == 'Embedding-Token/embeddings:0':
            w = np.concatenate((w[:4],w[104:-1]),axis=0)
        # if 'Transformer' in v_tf[i].name and 'kernel' in v_tf[i].name:
        #     w = np.transpose(w)
        op = tf.assign(v_tf[i],w)
        a1 = sess.run(op)
        w1 = sess.run(v_tf[i])
        print(v_tf[i].name,w.shape,w1.shape)
    else:
        print('----',v_tf[i].name)

saver = tf.train.Saver(tf.global_variables())
saver.save(sess, '/search/odin/guobk/data/model/pretrainCom/ckpt/model.ckpt-cor')
graph = tf.get_default_graph()
with tf.Session(graph=graph) as sess:
    bert.save_weights_as_checkpoint('/search/odin/guobk/data/model/pretrainCom/ckpt/model.ckpt-cor1')