export CUDA_VISIBLE_DEVICES=2
nohup python -u simbert.py >> log/train.log 2>&1 &

bert_model=chinese_simbert_L-4_H-312_A-12
path_model=/search/odin/guobk/data/my_simbert_l4
mkdir $path_model
nohup python -u simbert.py $bert_model $path_model >> log/train-l4.log 2>&1 &