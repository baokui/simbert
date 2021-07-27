from sentEmb import simbert_emb
import os
import json
def getContent():
    '''
    SQL数据库
    host: mt.tugele.rds.sogou
    port: 3306
    user: tugele_new
    password: tUgele2017OOT

    表名字 ns_flx_wisdom_words_new
    :return:
    '''
    import pymysql
    conn = pymysql.connect(
        host='mt.tugele.rds.sogou',
        user='tugele_new',
        password='tUgele2017OOT',
        charset='utf8',
        port  = 3306,
        # autocommit=True,    # 如果插入数据，， 是否自动提交? 和conn.commit()功能一致。
    )
        # ****python, 必须有一个游标对象， 用来给数据库发送sql语句， 并执行的.
        # 2. 创建游标对象，
    cur = conn.cursor()
    # 4). **************************数据库查询*****************************
    # sqli = 'SELECT * FROM tugele.ns_flx_wisdom_words_new'
    sqli = 'SELECT a.id,a.content,a.isDeleted,a.status FROM (tugele.ns_flx_wisdom_words_new a)'
    cur.execute('SET NAMES utf8mb4')
    cur.execute("SET CHARACTER SET utf8mb4")
    cur.execute("SET character_set_connection=utf8mb4")
    result = cur.execute(sqli)  # 默认不返回查询结果集， 返回数据记录数。
    info = cur.fetchall()  # 3). 获取所有的查询结果
    # print(info)
    # print(len(info))
    # 4. 关闭游标
    cur.close()
    # 5. 关闭连接
    conn.close()
    S = [[str(info[i][0]),info[i][1]] for i in range(len(info)) if info[i][2]==0 and info[i][3]==1]
    return S
def emb0():
    path_target = "/search/odin/guobk/data/search/juku/"
    model = simbert_emb()
    D = getContent()
    S = [d[1] for d in D]
    V = model.emb(S)
    Docs = {D[i][0]:D[i][1] for i in range(len(D))}
    Vecs = {D[i][0]:[float(t) for t in list(V[i])] for i in range(len(D))}
    with open(os.path.join(path_target,'Docs.json'),'w') as f:
        json.dump(Docs,f,ensure_ascii=False,indent=4)
    with open(os.path.join(path_target,'Vecs.json'),'w') as f:
        json.dump(Vecs,f,ensure_ascii=False,indent=4)
def emb1():
    from bson import ObjectId
    model = simbert_emb()
    path_target = "/search/odin/guobk/data/search/dabaigou/"
    with open(os.path.join(path_target,'sorted-all-filter2-rand-0.95.txt'),'r') as f:
        D = f.read().strip().split('\n')
    D = [d.split('\t') for d in D]
    for i in range(len(D)):
        D[i].insert(0,str(ObjectId()))
    S = [d[1] for d in D]
    V = model.emb(S)
    Docs = {D[i][0]:D[i][1] for i in range(len(D))}
    Vecs = {D[i][0]:[float(t) for t in list(V[i])] for i in range(len(D))}
    with open(os.path.join(path_target,'Docs.json'),'w') as f:
        json.dump(Docs,f,ensure_ascii=False,indent=4)
    with open(os.path.join(path_target,'Vecs.json'),'w') as f:
        json.dump(Vecs,f,ensure_ascii=False,indent=4)
    
def emb2():
    from bson import ObjectId
    model = simbert_emb()
    path_target = "/search/odin/guobk/data/search/prose/"
    with open(os.path.join(path_target,'content-0-dedup0-post15.txt'),'r') as f:
        D = f.read().strip().split('\n')
    D = [d.split('\t') for d in D]
    D = [d for d in D if len(d[1])>=5 and len(d[1])<=50]
    S = [d[1] for d in D]
    V = model.emb(S)
    Docs = {D[i][0]:D[i][1] for i in range(len(D))}
    Vecs = {D[i][0]:[float(t) for t in list(V[i])] for i in range(len(D))}
    with open(os.path.join(path_target,'Docs.json'),'w') as f:
        json.dump(Docs,f,ensure_ascii=False,indent=4)
    with open(os.path.join(path_target,'Vecs.json'),'w') as f:
        json.dump(Vecs,f,ensure_ascii=False,indent=4)

def emb3():
    model = simbert_emb()
    import random
    path_target = "/search/odin/guobk/data/search/query/"
    path_data = '/search/odin/guobk/data/vpaActive/20210713'
    D = []
    for i in range(5):
        with open(os.path.join(path_data,'part-0000'+str(i)),'r') as f:
            D.extend(f.read().strip().split('\n'))
    random.shuffle(D)
    D = [d.split('\t') for d in D]
    D = D[:1000]
    S = [d[1] for d in D]
    V = model.emb(S)
    Docs = {D[i][0]:D[i][1] for i in range(len(D))}
    Vecs = {D[i][0]:[float(t) for t in list(V[i])] for i in range(len(D))}
    with open(os.path.join(path_target,'Docs.json'),'w') as f:
        json.dump(Docs,f,ensure_ascii=False,indent=4)

def test_search():
    path_query = "/search/odin/guobk/data/search/query/"
    path_docs = ["/search/odin/guobk/data/search/juku/","/search/odin/guobk/data/search/prose/","/search/odin/guobk/data/search/dabaigou/"]
    tags = ['juku','meiwen','dabaigou']
    with open(os.path.join(path_query,'Docs.json'),'r') as f:
        Q = json.load(f)
    with open(os.path.join(path_query,'Vecs.json'),'r') as f:
        V_q = json.load(f)
    Ids_Q = [k for k in Q]
    V_q = [V_q[k] for k in Ids_Q]
    V_q = np.array(V_q)
    D = []
    V_d = []
    Ids_D = []
    for i in range(len(path_docs)):
        with open(os.path.join(path_docs[i],'Docs.json'),'r') as f:
            D.append(json.load(f))
        with open(os.path.join(path_docs[i],'Vecs.json'),'r') as f:
            V_d.append(json.load(f))
    Ids_D = [[k for k in d] for d in D]
    for i in range(len(Ids_D)):
        V_d[i] = np.array([V_d[i][k] for k in Ids_D[i]])
    Score = [V_q.dot(np.transpose(d)) for d in V_d]
    Idx = [np.argsort(-s,axis=-1) for s in Score]
    R = []
    for j in range(len(Q)):
        r = {'id':Ids_Q[j],'input':Q[Ids_Q[j]]}
        for i in range(len(D)):
            idx = Idx[i][j][:10]
            c = [D[i][Ids_D[i][ii]] for ii in idx]
            s = [Score[i][j][ii] for ii in idx]
            res = ['%0.4f'%s[k]+'\t'+c[k] for k in range(len(c))]
            r[tags[i]] = res
        R.append(r)
    with open(os.path.join(path_query,'test.json'),'w') as f:
        json.dump(R,f,ensure_ascii=False,indent=4)
    
    R1 = []
    for r in R:
        c0 = r['juku']
        c1 = r['meiwen']
        c0 = [t.split('\t') for t in c0]
        c1 = [t.split('\t') for t in c1]
        c0 = [(float(t[0]),'jk',t[1]) for t in c0]
        c1 = [(float(t[0]),'mw',t[1]) for t in c1]
        c1 = [(t[0]-0.05,t[1],t[2]) for t in c1]
        c = c0+c1
        c = sorted(c,key=lambda x:-x[0])
        c = ['\t'.join(['%0.4f'%t[0],t[1],t[2]]) for t in c]
        R1.append({'id':r['id'],'input':r['input'],'res':c})    
    with open(os.path.join(path_query,'test1.json'),'w') as f:
        json.dump(R1,f,ensure_ascii=False,indent=4)
        