from bert4keras.models import build_transformer_model
from bert4keras.layers import *
def apply_main_layers(q,k,v, model,index='cross'):
    """BERT的主体是基于Self-Attention的模块
    顺序：Att --> Add --> LN --> FFN --> Add --> LN
    """
    x = [q,k,v]
    z = None
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
    feed_forward_name = 'Transformer-%d-FeedForward' % index
    attention_mask = model.compute_attention_bias(index)
    # Self Attention
    xi, arguments = x[0], {'a_bias': None}
    if attention_mask is not None:
        arguments['a_bias'] = True
        x.append(attention_mask)
    x = model.apply(
        inputs=x,
        layer=MultiHeadAttention,
        arguments=arguments,
        heads=model.num_attention_heads,
        head_size=model.attention_head_size,
        out_dim=model.hidden_size,
        key_size=model.attention_key_size,
        kernel_initializer=model.initializer,
        name=attention_name
    )
    x = model.apply(
        inputs=x,
        layer=Dropout,
        rate=model.dropout_rate,
        name='%s-Dropout' % attention_name
    )
    x = model.apply(
        inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
    )
    x = model.apply(
        inputs=model.simplify([x, z]),
        layer=LayerNormalization,
        conditional=(z is not None),
        hidden_units=model.layer_norm_conds[1],
        hidden_activation=model.layer_norm_conds[2],
        hidden_initializer=model.initializer,
        name='%s-Norm' % attention_name
    )
    self = model
    # Feed Forward
    xi = x
    x = self.apply(
        inputs=x,
        layer=FeedForward,
        units=self.intermediate_size,
        activation=self.hidden_act,
        kernel_initializer=self.initializer,
        name=feed_forward_name
    )
    x = self.apply(
        inputs=x,
        layer=Dropout,
        rate=self.dropout_rate,
        name='%s-Dropout' % feed_forward_name
    )
    x = self.apply(
        inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
    )
    x = self.apply(
        inputs=self.simplify([x, z]),
        layer=LayerNormalization,
        conditional=(z is not None),
        hidden_units=self.layer_norm_conds[1],
        hidden_activation=self.layer_norm_conds[2],
        hidden_initializer=self.initializer,
        name='%s-Norm' % feed_forward_name
    )
    return x

class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        alpha = 0.5
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return alpha*loss1 + (1-alpha)*loss2
    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs[:4]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _, = inputs[:4]
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss
    def compute_loss_of_similarity_cross(self, inputs, mask=None):
        embA, embA_att = inputs[-2:]
        similarities = K.dot(embA, K.transpose(embA_att))  # 相似度矩阵
        y_true = K.eye(K.shape(embA)[0])
        # similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
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
def cross_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    labels = K.eye(K.shape(y_pred)[0])
    y_true = K.cast(labels, K.floatx())
    # 计算相似度
    similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
    similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
    # similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
    similarities = similarities * 30  # scale
    loss = K.categorical_crossentropy(
        y_true, similarities, from_logits=True
    )
    return K.mean(loss)

def create_model(config_path, checkpoint_path, keep_tokens):
    bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False
    )
    outputs, count = [], 0
    while True:
        try:
            output = bert.model.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break
    outputA = Lambda(lambda x: x[::2])(bert.model.outputs[0])#取偶数行，即取A句的featureA
    outputB = Lambda(lambda x: x[1::2])(outputs[-1])#取奇数行，即取B句的featureB
    outputA = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputA)
    outputB = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputB)

    queryEmb = keras.layers.GlobalAveragePooling1D()(outputs[-1])
    queryEmb = Lambda(lambda x: K.expand_dims(x,-2))(queryEmb)
    output = apply_main_layers(queryEmb, outputB, outputB,bert,index=len(outputs))
    outputA_att = Lambda(lambda x: K.squeeze(x,axis=1))(output)
    outputA_att = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputA_att)
    encoder = keras.models.Model(bert.model.inputs, [queryEmb, outputB])
    #seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])

    #outputs = TotalLoss([2, 3])(bert.model.inputs + bert.model.outputs + [outputA,outputA_att])
    outputs = [outputA,outputA_att]
    outputs = Lambda(lambda x: K.concatenate(x, axis=0))(outputs)

    model = keras.models.Model(bert.model.inputs, outputs)

    # AdamW = extend_with_weight_decay(Adam, 'AdamW')
    # optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
    # model.compile(optimizer=optimizer)
    # model.summary()
    return encoder, model