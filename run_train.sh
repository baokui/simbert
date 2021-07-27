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