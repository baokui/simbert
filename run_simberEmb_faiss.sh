CUDA_VISIBLE_DEVICES="0"
path_source="/search/odin/guobk/data/faiss_search/secondCorpus2/Docs.json"
path_target="/search/odin/guobk/data/faiss_search/secondCorpus2/"
path_model="/search/odin/guobk/data/my_simbert_l4/model_269.h5"
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
checkpoint_path="None"
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
python -u simbertEmb_faiss.py $path_source $path_target $path_model $config_path $checkpoint_path $dict_path >> log/faiss-secondCorpus2.log 2>&1 &