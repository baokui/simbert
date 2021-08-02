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

export CUDA_VISIBLE_DEVICES=0
corpus_path='/search/odin/guobk/data/Tab3_train/Q-all-0726.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
path_model='/search/odin/guobk/data/my_simbert_l4_sim-pretrain-mlmcse'
init_ckpt='/search/odin/guobk/data/model/pretrainCom/ckpt/model.ckpt-595914'
mkdir $path_model
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt >> log/train-l4-pretrain-mlmcse.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
corpus_path='/search/odin/guobk/data/Tab3_train/Q-all-0726.txt'
bert_model='chinese_simbert_L-4_H-312_A-12'
path_model='/search/odin/guobk/data/my_simbert_l4_sim-pretrain-roberta'
init_ckpt='/search/odin/guobk/data/model/pretrain/bert_model.ckpt'
mkdir $path_model
nohup python -u simbert_sim.py $corpus_path $bert_model $path_model $init_ckpt >> log/train-l4-pretrain-roberta.log 2>&1 &
