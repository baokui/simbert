from __future__ import print_function
import json
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.layers import *
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
from utils import create_model,apply_main_layers
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras.utils import multi_gpu_model
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
save_dir='/search/odin/guobk/data/model/bert_cross'
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
batch_size=64
gpus=2
nb_epochs=4
steps_per_epoch = 30000
maxlen=64
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)
def read_corpus():
    """读取语料，每行一个json
    """
    while True:
        with open(corpus_path) as f:
            for l in f:
                yield json.loads(l)
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


# encoder, model = create_model(config_path, init_ckpt, keep_tokens)
# encoder.summary()
# # encoder = keras.models.load_model(init_ckpt,compile = False)
# # encoder.compile(loss=cross_loss, optimizer=Adam(1e-5))
# checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
#                                    verbose=1, save_weights_only=False, period=1)
# train_generator = data_generator(read_corpus(), batch_size*gpus)
# # train_generator = data_generator(train_token_ids, batch_size*gpus)

# parallel_encoder = multi_gpu_model(model, gpus=gpus)
# parallel_encoder.compile(loss=cross_loss,
#                        optimizer=Adam(1e-5))
# encoder.save(os.path.join(save_dir,'model_init.h5'))
# parallel_encoder.fit(
#     train_generator.forfit(), steps_per_epoch=steps_per_epoch, epochs=nb_epochs,callbacks=[checkpointer]
# )
# encoder.save(os.path.join(save_dir,'model_final.h5'))
train_generator = data_generator(read_corpus(), batch_size*gpus)
iter = train_generator.forfit()
x,y = next(iter)



def cross_loss(y_true,y_pred):
    outputs = y_pred
    #outputB = Lambda(lambda x: x[1::2])(outputs)#取奇数行，即取B句的featureB
    outputB = outputs[1::2]
    #outputB = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputB)
    outputB = K.l2_normalize(outputB, axis=1)
    #queryEmb0 = Lambda(lambda x: x[::2])(outputs)
    queryEmb0 = outputs[::2]
    queryEmb = K.mean(queryEmb0,axis=1)
    # queryEmb = keras.layers.GlobalAveragePooling1D()(queryEmb0)
    #queryEmb = Lambda(lambda x: K.expand_dims(x,-2))(queryEmb)
    queryEmb = K.expand_dims(queryEmb,-2)
    output = apply_main_layers(queryEmb, outputB, outputB,bert,index=4)
    # outputA_att = Lambda(lambda x: K.squeeze(x,axis=1))(output)
    outputA_att = K.squeeze(output,axis=1)
    # outputA_att = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputA_att)
    outputA_att = K.l2_normalize(outputA_att,axis=1)
    y_predA = outputA_att
    # y_predB = Lambda(lambda x: K.squeeze(x,axis=1))(queryEmb)
    return queryEmb
    y_predB = K.squeeze(queryEmb,axis=1)
    b = K.cast(K.shape(y_predA)[0],dtype='int32')
    # 构造标签
    labels = K.eye(b)
    y_true = K.cast(labels, K.floatx())
    # 计算相似度
    similarities = K.dot(y_predA, K.transpose(y_predB))  # 相似度矩阵
    # similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
    similarities = similarities * 30  # scale
    loss = K.categorical_crossentropy(
        y_true, similarities, from_logits=True
    )
    return K.mean(loss)

bert, encoder = create_model(config_path, init_ckpt, keep_tokens)
encoder.summary()
encoder.compile(loss=cross_loss,
                       optimizer=Adam(1e-5))
parallel_encoder = multi_gpu_model(encoder, gpus=gpus)
parallel_encoder.compile(loss=cross_loss,
                       optimizer=Adam(1e-5))
encoder.save(os.path.join(save_dir,'model_init.h5'))
train_generator = data_generator(read_corpus(), batch_size*gpus)
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=False, period=1)
parallel_encoder.fit(
    train_generator.forfit(), steps_per_epoch=steps_per_epoch, epochs=nb_epochs,callbacks=[checkpointer]
)
encoder.save(os.path.join(save_dir,'model_final.h5'))