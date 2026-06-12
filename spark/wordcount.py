"""
WordCount 示例 —— Spark on K8s 作业提交验证用
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("WordCount") \
    .getOrCreate()

# 用豆瓣电影摘要作为示例文本
sample_texts = [
    "肖申克的救赎 希望 自由 救赎",
    "霸王别姬 京剧 爱情 背叛",
    "阿甘正传 跑步 人生 坚持",
]

lines = spark.sparkContext.parallelize(sample_texts)
word_counts = (
    lines.flatMap(lambda line: line.split())
         .map(lambda word: (word, 1))
         .reduceByKey(lambda a, b: a + b)
         .sortBy(lambda x: x[1], ascending=False)
)

print("Top 10 words:")
for word, count in word_counts.take(10):
    print(f"  {word}: {count}")

spark.stop()
