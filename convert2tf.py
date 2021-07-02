'''
cd ../SimCSE-1/
python keras_to_tensorflow.py \
    --input_model="/search/odin/guobk/data/my_simbert_l4/encoder_269.h5" \
    --output_model="/search/odin/guobk/data/my_simbert_l4/encoder_269.pb"
'''
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path_export_model,version = '/search/odin/guobk/data/my_simbert_l4/pbmodel/','0'
######################################################################################
# 程序开始时声明
tf.reset_default_graph()
sess = tf.Session()
# 读取得到的pb文件加载模型
with gfile.FastGFile("/search/odin/guobk/data/my_simbert_l4/encoder_269.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # 把图加到session中
    tf.import_graph_def(graph_def, name='')
    # 获取当前计算图
graph = tf.get_default_graph()
tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
# 从图中获输出那一层
#pred = graph.get_tensor_by_name("dense_73/Tanh:0")
pred = graph.get_tensor_by_name("Pooler-Dense/BiasAdd:0")
inputToken = graph.get_tensor_by_name("Input-Token:0")
inputSegment = graph.get_tensor_by_name("Input-Segment:0")
inputs = [inputToken,inputSegment]
# 保存为pb文件
sess.run(tf.global_variables_initializer())
builder = tf.saved_model.builder.SavedModelBuilder(path_export_model + "/" + version)
y = pred
#y = graph.get_tensor_by_name("Transformer-6-FeedForward-Norm/add_1:0")
signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'feat_index': inputToken,'feat_index1':inputSegment},
                                                                     outputs={'scores': y})
builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature,tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
builder.save()
print(path_export_model + "/" + version)

######################################################################################
# docker 部署tf-serving
'''
source=/search/odin/guobk/data/my_simbert_l4/pbmodel/
model=simbertSearch
target=/models/$model
ps -ef | grep 8501|grep -v grep | awk '{print "kill -9 "$2}'|sh
sudo docker run -p 8501:8501 --mount type=bind,source=$source,target=$target -e MODEL_NAME=$model -t tensorflow/serving >> ./log/tfserving-cpu-$model.log 2>&1 &
curl http://localhost:8501/v1/models/$model/versions/0
curl http://localhost:8501/v1/models/$model #查看模型所有版本服务状态
curl http://localhost:8501/v1/models/$model/metadata #查看服务信息，输入大小等

'''

######################################################################################
# 验证生成环境结果
import requests
import keras
from model import get_model
from bert4keras.snippets import sequence_padding
def emb(encoder,Sent):
    X, S = [], []
    x, s = tokenizer.encode(Sent)
    X.append(x)
    S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return Z,X,S
url = 'http://10.160.25.112:8501/v1/models/simbertSearch:predict'
# url = 'http://tensorflow-bert-semantic.thanos.sogou/v1/models/bert_semantic_simcse:predict'
# url = 'http://yunbiaoqing-tensorflow.thanos-lab.sogou/v1/models/bert_semantic_simcse:predict'
# url = 'http://yunbiaoqing-tensorflow.thanos-lab.sogou/v1/models/bert_semantic_simcse:predict'
path_model = '/search/odin/guobk/data/my_simbert_l4/model_269.h5'
bert_model = 'chinese_simbert_L-4_H-312_A-12'
model, seq2seq, encoder,tokenizer = get_model(bert_model)
model.load_weights(path_model)
# encoder = keras.models.load_model('/search/odin/guobk/data/my_simbert_l4/encoder_269.h5',compile = False)
sent = "可别昧着良心做事，"
V0,X,S = emb(encoder,sent)
V0 = V0[0]
x0 = [int(i) for i in list(X[0])]
x1 = [0 for i in range(len(x0))]
feed_dict = {'instances': [{'feat_index': x0, 'feat_index1': x1}]}
r = requests.post(url,json=feed_dict)
V1 = r.json()['predictions'][0]
V1 = np.array(V1)
V1 = V1/np.sqrt(V1.dot(V1))
mse = np.mean([(V0[i]-V1[i])**2 for i in range(len(V0))])
