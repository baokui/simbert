export CUDA_VISIBLE_DEVICES="3"
path_model=/search/odin/guobk/data/my_simbert_l4/model_269.h5
bert_model=chinese_simbert_L-4_H-312_A-12
tag=rec_simbert_l4
path_docs=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json
path_queries=/search/odin/guobk/data/Tab3_test/Q-20210629.json
path_target=/search/odin/guobk/data/Tab3_test/Q-20210629-simbert-l4.json
maxQ=200
python -u test_search.py $models $path_models $path_docs $path_queries $path_target $maxQ >> log/test-0629.log 2>&1 &