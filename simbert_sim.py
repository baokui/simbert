#! -*- coding: utf-8 -*-
# SimBERT训练代码
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

from __future__ import print_function
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
from sklearn.metrics import roc_auc_score
import os
import random
import sys
# 基本信息
maxlen = 32
batch_size = 128
# steps_per_epoch = 30000
epochs = 20
# alpha = 0.0001
# corpus_path = '/search/odin/guobk/data/Tab3_train/Q-all-0726.txt'
# bert_model = 'chinese_simbert_L-4_H-312_A-12'
# path_model = '/search/odin/guobk/data/my_simbert_l4_sim'
corpus_path,bert_model,path_model,init_ckpt,config_path,dict_path,test_path,alpha,nb_train_examples = sys.argv[1:10]
if len(sys.argv)<10:
    train_gen = 0
else:
    train_gen = int(sys.argv[10])
steps_per_epoch = int(int(nb_train_examples)*2/batch_size)
alpha = float(alpha)
# bert配置
# config_path = '/search/odin/guobk/data/model/{}/bert_config.json'.format(bert_model)
# checkpoint_path = '/search/odin/guobk/data/model/{}/bert_model.ckpt'.format(bert_model)
checkpoint_path = init_ckpt
# dict_path = '/search/odin/guobk/data/model/{}/vocab.txt'.format(bert_model)


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

with open(test_path,'r') as f:
    D = json.load(f)
Sents = [d['input'] for d in D]
for d in D:
    Sents.extend(d['pos'])
    Sents.extend(d['neg'])
Sents = list(set(Sents))
print('test set: %d'%len(Sents))

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

def read_corpus():
    """读取语料，每行一个json
    """
    while True:
        with open(corpus_path) as f:
            for l in f:
                yield json.loads(l)

with open(corpus_path,'r') as f:
    S = f.read().strip().split('\n')
# TrnData = [json.loads(f) for f in S]

def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, d in self.sample(random):
            text, synonyms = d['input'], d['click']
            synonyms = [text] + synonyms
            np.random.shuffle(synonyms)
            text, synonym = synonyms[:2]
            while text==synonym:
                np.random.shuffle(synonyms)
                text, synonym = synonyms[:2]
            text, synonym = truncate(text), truncate(synonym)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, segment_ids = tokenizer.encode(
                text, synonym, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class data_generator1(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator1, self).__init__(*args, **kwargs)
        self.some_samples = []
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, d in self.sample(random):
            text, synonyms = d['input'], d['click']
            np.random.shuffle(synonyms)
            synonym = synonyms[0]
            text, synonym = truncate(text), truncate(synonym)
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, segment_ids = tokenizer.encode(
                text, synonym, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                synonym, text, maxlen=maxlen * 2
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return alpha*loss1 + (1-alpha)*loss2
    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss
    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels


# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs)
model = keras.models.Model(bert.model.inputs, outputs)

AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.summary()


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


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20):
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


def just_show():
    """随机观察一些样本的效果
    """
    # some_samples = train_generator.some_samples
    # S = [np.random.choice(some_samples) for i in range(3)]
    S = random.sample(TrnData, k=10)
    for s in S:
        try:
            print('###########################')
            print('------------------')
            print(u'原句子：%s' % s['input'])
            print(u'同义句子：')
            r = gen_synonyms(s['click'], 10, 10)
            for rr in r:
                print(rr)
            print('------------------')
            print(u'原句子：%s' % s['click'][0])
            print(u'同义句子：')
            r = gen_synonyms(s['click'][0], 10, 10)
            for rr in r:
                print(rr)
        except:
            pass
def getAcc(labels,preds,thr):
    y = [int(t>=thr) for t in preds]
    n = sum([y[i]==labels[i] for i in range(len(y))])
    return n/len(y)

def eval(epoch):
    V_d = emb(encoder,Sents)
    D_v = {Sents[i]:V_d[i] for i in range(len(Sents))}
    labels = []
    preds = []
    for d in D:
        labels.extend([1]*len(d['pos']))
        labels.extend([0]*len(d['neg']))
        v0 = D_v[d['input']]
        v1 = [D_v[t] for t in d['pos']] + [D_v[t] for t in d['neg']]
        v1 = np.array(v1)
        s = v1.dot(v0)
        preds.extend(list(s))
    nb_pos = sum(labels)
    nb_neg = len(labels) - nb_pos
    auc = roc_auc_score(labels,preds)
    acc = getAcc(labels,preds,thr=0.5)
    s = ['model evaluation of epoch {} ...'.format(epoch)]
    s.append('num of pos and neg: %d, %d'%(nb_pos,nb_neg))
    s.append('auc: %0.4f'%auc)
    s.append('acc: %0.4f'%acc)
    print('\n'.join(s))

class Evaluate(keras.callbacks.Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(os.path.join(path_model,'latest_model.weights'))
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(os.path.join(path_model,'latest_model.weights'))
        # 演示效果
        # just_show()
        eval(epoch)

def test():
    model.load_weights(os.path.join(path_model,'latest_model.weights'))
    just_show()
if __name__ == '__main__':
    if train_gen==0:
        print('train_generator with simple mode')
        train_generator = data_generator(read_corpus(), batch_size)
    else:
        print('train_generator with query-doc mode')
        train_generator = data_generator1(read_corpus(), batch_size)
    evaluator = Evaluate()
    checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(path_model, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=True, period=1)
    eval('init')
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[checkpointer,evaluator]
    )

else:
    pass
    # model.load_weights('./latest_model.weights')
