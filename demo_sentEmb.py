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
    Vecs = {D[i][0]:list(V[i]) for i in range(len(D))}
    with open(os.path.join(path_target,'Docs.json'),'w') as f:
        json.dump(Docs,f,ensure_ascii=False,indent=4)
    with open(os.path.join(path_target,'Vecs.json'),'w') as f:
        json.dump(Vecs,f,ensure_ascii=False,indent=4)
    

