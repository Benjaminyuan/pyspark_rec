
from pyspark import SparkContext
from pyspark import SparkConf
import math
import json
import sys
def CreatSparkContext():
    sparkConf=SparkConf() .setAppName("WordCounts").set("spark.ui.showConsoleProgress","false")
    sc=SparkContext(conf=sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    SetPath(sc)
    return(sc)
def SetLogger(sc):
    logger=sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
def SetPath(sc):
    global Path
    if sc.master[0:5]=="local":
        Path="file:/home/hduser/pythonwork/PythonProject/"
    else:
        Path="hdfs://localhost:9000/user/hduser/"
def prepardata(sc):
    rawUserData=sc.textFile(Path+"data/u.data")
    rawRatings=rawUserData.map(lambda line:line.split("\t")[:2])
    ratingsRDD=rawRatings.map(lambda x:(x[0],x[1]))
    return(ratingsRDD)
def calculate(x1,x2):
    count1=[ ]
    count2=[ ]
    for x in x1:
        count1.append(x)
    for x in x2:
        count2.append(x)
    count=len(count1)*len(count2)*1.0
    a=set(count1)
    b=set(count2)
    c=a&b
    bothcount=len(c)
    w=bothcount/math.sqrt(count)
    return w
def find(x,list_item):
    for i in list_item:
        if i==x:
            return True
    return False
def get(x,k):
    x.sort(key=lambda x:x[1],reverse=True)
    return x[:k]
def recommend(W,user_item,user_id,k):
    user_see_item=user_item.filter(lambda x:x[0]==user_id).map(lambda x:x[1]).collect()[0]
    sim_item=W.filter(lambda x:find(x[0],user_see_item)).map(lambda x:x[1])
    recommend_item=sim_item.flatMap(lambda x:get(x,k)).reduceByKey(lambda x,y:x+y)
    recommend_item=recommend_item.sortBy(lambda x:x[1],False)
    return recommend_item.take(k)
if __name__=="__main__":
    if len(sys.argv)!=3:
        print("wrong input")
        exit(-1)
    sc=CreatSparkContext()
    print("prepare data")
    ratingsRDD=prepardata(sc)
    user_item=ratingsRDD.groupByKey().map(lambda x:(x[0],list(x[1])))
    print("got user_item")
    RDD=ratingsRDD.map(lambda x:(x[1],x[0]))
    item_user=RDD.groupByKey().map(lambda x:(x[0],list(x[1])))
    print("got item_user")
    print("item_user---item_user")
    item_user=item_user.cartesian(item_user).filter(lambda x:x[0][0]!=x[1][0])
    print("calculate")
    w=item_user.map(lambda x:(x[0][0],x[1][0],calculate(x[0][1],x[1][1])))
    W=w.map(lambda x:(x[0],(x[1],x[2])))
    W=W.groupByKey().map(lambda x:(x[0],list(x[1])))
    print("got similiar table")
    print("recommending.......")
    user_id=u'200'
    recommend=recommend(W,user_item,user_id,5)
    print("Result:")
    for r in recommend:
        print(r)
