"""
A-3: 性能对比 —— Pandas vs PySpark (Amdahl 分析)
"""
import time
import pandas as pd
from pyspark.sql import SparkSession

DATA = r"D:\大三下\云计算\douban_movies.csv"

# ===== Pandas 单机 =====
print("=" * 50)
print("Pandas 单机版本")
print("=" * 50)
t1 = time.time()
pdf = pd.read_csv(DATA, encoding="utf-8")
pdf_clean = pdf.dropna(subset=["rating_score"])
pdf_clean["year"] = pdf_clean["year"].fillna(pdf_clean["year"].mean())
pdf_clean["decade"] = (pdf_clean["year"] // 10 * 10).astype(int)

result_pd = pdf_clean.groupby("decade").agg(
    movie_count=("rating_score", "count"),
    avg_rating=("rating_score", "mean")
).reset_index()
t2 = time.time()
pd_time = t2 - t1
print(f"Pandas 执行时间: {pd_time:.2f}s")
print(result_pd.head(10))
print(f"数据量: {len(pdf)} 行")

# ===== PySpark 单 Executor =====
print("\n" + "=" * 50)
print("PySpark 版本 (local[2])")
print("=" * 50)

spark = SparkSession.builder \
    .appName("PerfCompare") \
    .config("spark.master", "local[2]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

t3 = time.time()
sdf = spark.read.csv(DATA, header=True, inferSchema=True, encoding="UTF-8")
sdf_clean = sdf.filter(sdf.rating_score.isNotNull())
sdf_clean.createOrReplaceTempView("movies")
result_spark = spark.sql("""
    SELECT FLOOR(year/10)*10 AS decade,
           COUNT(*) AS movie_count,
           ROUND(AVG(rating_score), 2) AS avg_rating
    FROM movies WHERE year IS NOT NULL AND year > 0
    GROUP BY FLOOR(year/10)*10 ORDER BY decade
""")
result_spark.collect()
t4 = time.time()
spark_time = t4 - t3
print(f"PySpark 执行时间: {spark_time:.2f}s")
result_spark.show(10, truncate=False)

# ===== 加速比分析 =====
print("\n" + "=" * 50)
print("加速比 & Amdahl 分析")
print("=" * 50)
speedup = pd_time / spark_time
print(f"Pandas 单机: {pd_time:.2f}s")
print(f"PySpark [2]: {spark_time:.2f}s")
print(f"加速比 S = {speedup:.2f}x")

print("""
Amdahl 定律分析:
  S = 1 / [(1-f) + f/p]
  其中 f = 可并行比例, p = 并行度

  本次测试中:
  - 加速比未达线性的原因: 数据量小(118k 行, ~6MB),
    Spark 的启动开销(JVM, 序列化, 调度)远大于计算时间
  - 当数据量增大到 GB 级别时, 加速比会更接近线性
  - 串行部分(数据加载, 结果收集)限制了最大加速比

  结论: 对于小数据集, Pandas 更高效;
        Spark 优势在大规模分布式计算场景
""")

spark.stop()
