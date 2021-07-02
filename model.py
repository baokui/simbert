from __future__ import print_function
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
def get_model(bert_model):
    # bert配置
    config_path = '/search/odin/guobk/data/model/{}/bert_config.json'.format(bert_model)
    checkpoint_path = '/search/odin/guobk/data/model/{}/bert_model.ckpt'.format(bert_model)
    dict_path = '/search/odin/guobk/data/model/{}/vocab.txt'.format(bert_model)
    token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)
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
    # AdamW = extend_with_weight_decay(Adam, 'AdamW')
    # optimizer = AdamW(learning_rate=2e-6, weight_decay_rate=0.01)
    # model.compile(optimizer=optimizer)
    # model.summary()
    return model, seq2seq, encoder