from utils import create_model,cross_loss
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras.utils import multi_gpu_model
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
save_dir='/search/odin/guobk/data/model/bert_cross'
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
batch_size=64
gpus=2
steps_per_epoch = 30000
def read_corpus():
    """读取语料，每行一个json
    """
    while True:
        with open(corpus_path) as f:
            for l in f:
                yield json.loads(l)
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

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
encoder, model = create_model(config_path, checkpoint_path, keep_tokens)
encoder.summary()
encoder = keras.models.load_model(init_ckpt,compile = False)
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                                   verbose=1, save_weights_only=False, period=1)
train_generator = data_generator(read_corpus(), batch_size)
train_generator = data_generator(train_token_ids, batch_size*gpus)

parallel_encoder = multi_gpu_model(encoder, gpus=gpus)
parallel_encoder.compile(loss=cross_loss,
                       optimizer=Adam(1e-5))
encoder.save(os.path.join(save_dir,'model_init.h5'))
parallel_encoder.fit(
    train_generator.forfit(), steps_per_epoch=steps_per_epoch, epochs=nb_epochs,callbacks=[checkpointer]
)
encoder.save(os.path.join(save_dir,'model_final.h5'))
