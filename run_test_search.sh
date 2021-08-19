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


export CUDA_VISIBLE_DEVICES="1"
model='my_simbert_l4_sim'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
tag='rec_simbert_l4_sim-pretrain-mlmcse'
path_docs='/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json'
path_queries='/search/odin/guobk/data/Tab3_test/test-20210804.json'
path_target='/search/odin/guobk/data/Tab3_test/test-20210804.json'
checkpoint_path=None
maxQ=10000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$model.log 2>&1 &

export CUDA_VISIBLE_DEVICES="1"
model='my_simbert_l4_sim-pretrain-mlmcse'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
tag='rec_'$model
path_docs='/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Docs.json'
path_queries='/search/odin/guobk/data/Tab3_test/test-20210804.json'
path_target='/search/odin/guobk/data/Tab3_test/test-20210804.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config_re.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab_re.txt'
checkpoint_path=None
maxQ=10000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$model.log 2>&1 &

export CUDA_VISIBLE_DEVICES="1"
model='my_simbert_l4'
path_model='/search/odin/guobk/data/'$model'/model_269.h5'
path_data='/search/odin/guobk/data/Tab3_test/test-20210804-testdata.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
python -u eval.py $path_model $path_data $config_path $checkpoint_path $dict_path >> log/eval-$model.log 2>&1 &

export CUDA_VISIBLE_DEVICES="2"
model='my_simbert_l4_sim'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
path_data='/search/odin/guobk/data/Tab3_test/test-20210804-testdata.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
python -u eval.py $path_model $path_data $config_path $checkpoint_path $dict_path >> log/eval-$model.log 2>&1 &

export CUDA_VISIBLE_DEVICES="0"
model='my_simbert_l4'
path_model='/search/odin/guobk/data/'$model'/model_269.h5'
path_data='/search/odin/guobk/data/Tab3_test/test-20210804-testdata-5.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
python -u eval.py $path_model $path_data $config_path $checkpoint_path $dict_path >> log/eval-5-$model.log 2>&1 &

export CUDA_VISIBLE_DEVICES="1"
model='my_simbert_l4_sim'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
path_data='/search/odin/guobk/data/Tab3_test/test-20210804-testdata-5.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
python -u eval.py $path_model $path_data $config_path $checkpoint_path $dict_path >> log/eval-5-$model.log 2>&1 &

export CUDA_VISIBLE_DEVICES="3"
model='my_simbert_l4_sim-pretrain-mlmcse'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
path_data='/search/odin/guobk/data/Tab3_test/test-20210804-testdata.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config_re.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab_re.txt'
checkpoint_path=None
python -u eval.py $path_model $path_data $config_path $checkpoint_path $dict_path $test_path >> log/eval-$model-2.log 2>&1 &



################################3
export CUDA_VISIBLE_DEVICES="0"
model='my_simbert_l4_sim0809-flatnce'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
tag='rec_'$model
path_docs='/search/odin/guobk/data/vpaSupData/Docs-0809.json'
path_queries='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
path_target='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
maxQ=1000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$model.log 2>&1 &

################################3
export CUDA_VISIBLE_DEVICES="0"
model='my_simbert_l4_sim0809'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
tag='rec_'$model
path_docs='/search/odin/guobk/data/vpaSupData/Docs-0809.json'
path_queries='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
path_target='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
maxQ=1000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$model.log 2>&1 &

################################3
export CUDA_VISIBLE_DEVICES="0"
model='my_simbert_l4_sim-pretrain-mlmcse'
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
tag='rec_'$model
path_docs='/search/odin/guobk/data/vpaSupData/Docs-0809.json'
path_queries='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
path_target='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config_re.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab_re.txt'
checkpoint_path=None
maxQ=1000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$model.log 2>&1 &

######################
export CUDA_VISIBLE_DEVICES="0"
model='my_simbert_l4'
path_model='/search/odin/guobk/data/'$model'/model_269.h5'
tag='rec_'$model
path_docs='/search/odin/guobk/data/vpaSupData/Docs-0809.json'
path_queries='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
path_target='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
maxQ=1000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$model.log 2>&1 &


################################3
export CUDA_VISIBLE_DEVICES="0"
alpha=0.75
tag='alpha_'$alpha
model='my_simbert_l4_sim_alpha/'$tag
path_model='/search/odin/guobk/data/'$model'/latest_model.weights'
tag='rec_'$tag
path_docs='/search/odin/guobk/data/vpaSupData/Docs-0809.json'
path_queries='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
path_target='/search/odin/guobk/data/vpaSupData/Q-all-test-20210809-rec.json'
config_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/bert_config.json'
dict_path='/search/odin/guobk/data/model/chinese_simbert_L-4_H-312_A-12/vocab.txt'
checkpoint_path=None
maxQ=1000
python -u test_search.py $path_model $tag $path_docs $path_queries $maxQ $path_target $config_path $checkpoint_path $dict_path >> log/test-search-$tag.log 2>&1 &
