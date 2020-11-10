from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType
from scipy.linalg import norm
import numpy as np
import heapq
import csv
import math
import json
import sys
Path = "/home/yjw/movie_rec/moivelens/ml-latest-small/"
tag_map = {}

spark = SparkSession.builder.enableHiveSupport().getOrCreate()


def CreatSparkContext():
    sparkConf = SparkConf().setAppName("userTagCount").set("spark.ui.showConsoleProgress","false")
    sc = SparkContext(conf = sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    return(sc)
def SetLogger(sc):
    logger=sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

#读取聚合后的tagMap
def readTagMap(sc,file):
    tagFile = open(Path + file)
    for line in tagFile:
        tagList = line.split(" ")
        if len(tagList) < 2:
            continue
        tag_map[tagList[0]] = tagList[1]
    return tag_map
# 读取用户数据
def readMovieGenome(sc,tagMap,file):
    df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(Path + file).rdd
    #movieRDD = df.map(lambda x:(x[0],(x[1],x[2]))).groupByKey().map(lambda x : (x[0],list(x[1])))
    movieRDD = df.map(lambda x:(x[0],x[1],x[2]))
    # print(movieRDD.first())
    return movieRDD
#读取每部电影的电影评分
def readRatingsData(sc,file):
    rdd = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(Path + file).rdd
    #(userId,movieId,Ratings)
    userRDD = rdd.map(lambda x:(x[0],(x[1],x[2]))).groupByKey().map(lambda x : (x[0],list(x[1])))
    #(moiveId,userId,ratings)
    # ratingsRDD = rdd.map(lambda x:(x[1],(x[0],x[2]))).groupByKey().map(lambda x : (x[0],list(x[1])))
    ratingsRDD = rdd.map(lambda x:(x[1],x[0],x[2]))
    print(userRDD.take(4))
    print("first -----")
    print(ratingsRDD.take(5))
    return userRDD,ratingsRDD
# def calculate(x1,x2):
#     return w
def tranformtion(line):
    tagMap = tranformtion.tagMap
    if len(tagMap) == 0:
        print("empty tagMap")
        return 
    userTagMap = dict()
    tags = line[1]
    for tagPair in tags:
        tag = tagPair[0]
        if tag in userTagMap:
            userTagMap[tag] += tagPair[1]
        else:
            userTagMap[tag] = tagPair[1]
    # print(line[0],userTagMap)   
    return (line[0],userTagMap)
def simple_cosine_sim(a,b):

    res = 0
    abase = 0
    bbase = 0
    for key,a_value in a.items():
        res += a_value * b.get(key,0)
        abase += a_value * a_value
    for key,b_value in b.items():
        bbase += b_value * b_value
    if res == 0:
        return 0
    try:
        res = res / (np.sqrt(abase) * np.sqrt(bbase))
    except ZeroDivisionError:
        res = 0
    return res

def recommend(user_sim,user_item,user_id,k):
    print(user_sim.count())
    user_sim = user_sim.filter(lambda x :x[0] == user_id).map(lambda x:(x[1],x[2])).collect()
    # recommend_user = user_sim.map(lambda x:x[1]).sortBy(lambda x: x[1],False)
    print(user_sim)
    res = heapq.nlargest(3,user_sim,key=lambda x:x[1])
    user_sim.sort(key=lambda x : x[1],reverse = True)
    print(res)
    print(user_sim)

def recommend_two(user_tag_rdd,user_id,k,user_item):
    # 特征矩阵，先通过用户ID直接筛选掉其他的用户
    print("---get rec user tag data---")
    print("total user size: ",user_tag_rdd.count())
    print("searching.....")
    user_data = user_tag_rdd.filter(lambda x: x[0] == user_id).first()
    print(user_data)
    print("---count similarity---")
    user_sim = user_tag_rdd.filter(lambda x: x[0] != user_id).map(lambda x : (x[0],simple_cosine_sim(x[1],user_data[1]))).collect()
    print("--- sort ---")
    res_user = heapq.nsmallest(k,user_sim,lambda x:x[1])
    user_sim.sort(key=lambda x : x[1],reverse = False)
    print("--- result ---")
    print(res_user)
    
if __name__=="__main__":
    sc = spark.sparkContext
    print("prepare data")
    user_movie,movie_user = readRatingsData(sc,"ratings.csv")
    print("--------")
    tag_map = readTagMap(sc,"output.txt")
    tranformtion.tagMap = tag_map
    print("read tagMapFinish ---")
    movie_tag = readMovieGenome(sc,tag_map,"genome-scores.csv")

    movie_user_def_columns = ["movie_id","user_id","ratings"]
    movie_tag_def_columns = ["movie_id","tag_id","relevance"]

    movie_user_df = movie_user.toDF(movie_user_def_columns)
    movie_tag_df = movie_tag.toDF(movie_tag_def_columns)
    j1 = movie_user_df.join(movie_tag_df,on = "movie_id",how = "inner")
    movie_user_df = None
    movie_tag_df = None
    # j1.printSchema()
    # j1.show(3)
    user_tag_rdd = j1.rdd
    j1 = None
    user_tag_rdd = user_tag_rdd.\
        map(lambda x:(x.user_id, (x.tag_id, x.ratings * x.relevance)))\
        .groupByKey()\
        .map(lambda x: (x[0],list(x[1])))\
        .map(lambda x : tranformtion(x))
    # print(user_tag_rdd.count())
    # print("---userTag matrix---")
    # # 特征矩阵，完全聚合
    # user_tag_rdd = user_tag_rdd.cartesian(userTagRDD).filter(lambda x: x[0][0] != x[1][0])
    # print(user_tag_rdd.count())
    # print("---count similarity---")
    # # 计算相似度
    # sim = user_tag_rdd.map(lambda x :(x[0][0],x[1][0],simple_cosine_sim(x[0][1],x[1][1])))
    # user_tag_rdd = None
    # # sim = sim.map(lambda x:(x[0],(x[1],x[2])))
    # # sim = sim.groupByKey().map(lambda x:(x[0],list(x[1])))
    # # print(sim.take(5))
    # # 推荐
    # print("---start recommend---")
    # recommend(sim,user_movie,1,2)
    recommend_two(user_tag_rdd,4,10,user_movie)