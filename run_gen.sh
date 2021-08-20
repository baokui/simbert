CUDA_VISIBLE_DEVICES="0"
path_source="/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json"
path_target="/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-gen.json"
tags="ori,alpha_0,alpha_0.1,alpha_0.25,alpha_0.5,alpha_0.75"
path_model="/search/odin/guobk/data/my_simbert_l4/model_269.h5,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.1/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.25/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.5/latest_model.weights,/search/odin/guobk/data/my_simbert_l4_sim_alpha/alpha_0.75/latest_model.weights"
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
checkpoint_path="None"
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
python -u simbertEmb_gen.py $path_source $path_target $path_model $tags $config_path $checkpoint_path $dict_path >> log/faiss-secondCorpus2.log 2>&1 &
