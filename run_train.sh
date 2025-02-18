export CUDA_VISIBLE_DEVICES=2
nohup python -u simbert.py >> log/train.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
bert_model=chinese_simbert_L-4_H-312_A-12
path_model=/search/odin/guobk/data/my_simbert_l4
mkdir $path_model
nohup python -u simbert.py $bert_model $path_model >> log/train-l4.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
bert_model=chinese_simbert_L-6_H-384_A-12
path_model=/search/odin/guobk/data/my_simbert_l6
mkdir $path_model
nohup python -u simbert.py $bert_model $path_model >> log/train-l6.log 2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup python -u simbert_sim.py >> log/train-sim-l4.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
corpus_path='/search/odin/guobk/data/Tab3_train/Q-all-0726.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
path_model='/search/odin/guobk/data/my_simbert_l4_sim-pretrain-mlmcse'
init_ckpt='/search/odin/guobk/data/model/pretrainCom/ckpt/model.ckpt-cor1'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config_re.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab_re.txt'
mkdir $path_model
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path >> log/train-l4-pretrain-mlmcse2.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# corpus_path='/search/odin/guobk/data/Tab3_train/Q-all-0726.txt'
# bert_model='chinese_simbert_L-4_H-312_A-12'
# path_model='/search/odin/guobk/data/my_simbert_l4_sim-pretrain-roberta'
# init_ckpt='/search/odin/guobk/data/model/pretrain/bert_model.ckpt'
# mkdir $path_model
# nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path >> log/train-l4-pretrain-roberta2.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
path_model='/search/odin/guobk/data/my_simbert_l4_sim0809'
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
mkdir $path_model
test_path='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json'
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path $test_path >> log/train-l4-0809.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
path_model='/search/odin/guobk/data/my_simbert_l4_sim0809-flatnce'
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
mkdir $path_model
test_path='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json'
nohup python -u simbert_FlatNCE.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path $test_path >> log/train-l4-0809-flatnce.log 2>&1 &


##################
export CUDA_VISIBLE_DEVICES=0
path_root='/search/odin/guobk/data/my_simbert_l4_sim_alpha/'
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
alpha=0.5
tag='alpha_'$alpha
path_model=$path_root$tag
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
mkdir $path_model
test_path='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json'
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path $test_path $alpha >> log/train-sim_alpha-$alpha.log 2>&1 &

##################
export CUDA_VISIBLE_DEVICES=1
path_root='/search/odin/guobk/data/my_simbert_l4_sim_alpha/'
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
alpha=0.1
tag='alpha_'$alpha
path_model=$path_root$tag
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
mkdir $path_model
test_path='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json'
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path $test_path $alpha >> log/train-sim_alpha-$alpha.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
path_root='/search/odin/guobk/data/my_simbert_l4_sim_alpha/'
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
alpha=0.25
tag='alpha_'$alpha
path_model=$path_root$tag
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
mkdir $path_model
test_path='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json'
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path $test_path $alpha >> log/train-sim_alpha-$alpha.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
path_root='/search/odin/guobk/data/my_simbert_l4_sim_alpha/'
corpus_path='/search/odin/guobk/data/vpaSupData/Q-all-train-20210809.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
alpha=0.75
tag='alpha_'$alpha
path_model=$path_root$tag
init_ckpt='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
mkdir $path_model
test_path='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809.json'
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt $config_path $dict_path $test_path $alpha >> log/train-sim_alpha-$alpha.log 2>&1 &