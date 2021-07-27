export CUDA_VISIBLE_DEVICES="2"
path_model=/search/odin/guobk/data/my_simbert_l4/model_269.h5
bert_model=chinese_simbert_L-4_H-312_A-12
tag=rec_simbert_l4
path_docs=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json
path_queries=/search/odin/guobk/data/Tab3_test/Q-20210629.json
path_target=/search/odin/guobk/data/Tab3_test/Q-20210629.json
maxQ=200
python -u test_search.py $path_model $bert_model $tag $path_docs $path_queries $maxQ $path_target >> log/test-0629.log 2>&1 &


export CUDA_VISIBLE_DEVICES="2"
path_model=/search/odin/guobk/data/my_simbert_l4/model_269.h5
bert_model=chinese_simbert_L-4_H-312_A-12
tag=rec_simbert_l4
path_docs=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json
path_queries=/search/odin/guobk/data/vpaActive/Queries.json
path_target=/search/odin/guobk/data/vpaActive/Queries.json
maxQ=10000
python -u test_search.py $path_model $bert_model $tag $path_docs $path_queries $maxQ $path_target >> log/test-active.log 2>&1 &

export CUDA_VISIBLE_DEVICES="0"
path_model=/search/odin/guobk/data/my_simbert_l4_sim/latest_model.weights
bert_model=chinese_simbert_L-4_H-312_A-12
tag=rec_simbert_l4_sim
path_docs=/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json
path_queries=/search/odin/guobk/data/vpaActive/Queries.json
path_target=/search/odin/guobk/data/vpaActive/Queries.json
maxQ=10000
python -u test_search.py $path_model $bert_model $tag $path_docs $path_queries $maxQ $path_target >> log/test-active-sim.log 2>&1 &