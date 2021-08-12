path_source="/search/odin/guobk/data/faiss_search/secondCorpus/Docs.json"
path_target="/search/odin/guobk/data/faiss_search/secondCorpus/"
python -u simbertEmb_faiss.py $path_target $path_target >> log/faiss-secondCorpus.log 2>&1 &