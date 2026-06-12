"""
本地运行版本 —— 使用本地 douban_movies.csv
"""
import sys, os
os.chdir(r"D:\cloud_project\spark")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, stddev, min, max, desc, asc, row_number, floor
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType

spark = SparkSession.builder \
    .appName("DoubanMoviesAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

DATA_PATH = r"D:\大三下\云计算\douban_movies.csv"

# ============================
# A-1: 数据清洗
# ============================
print("=" * 60)
print("A-1: 数据清洗")
print("=" * 60)

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True, encoding="UTF-8")
total = df.count()
print(f"\n1. 加载完成，原始数据行数: {total}")
print("\nSchema:")
df.printSchema()
print("\n前 5 行:")
df.show(5, truncate=False)

print("\n2. 各字段缺失值统计:")
for c in df.columns:
    null_count = df.filter(col(c).isNull()).count()
    pct = null_count / total * 100
    print(f"  {c}: {null_count} 缺失 ({pct:.2f}%)")

# 策略A: 删除 rating_score 缺失
cleaned = df.filter(col("rating_score").isNotNull())
after_a = cleaned.count()
print(f"\n3a. 删除 rating_score 缺失行: {total} -> {after_a} (删除 {total - after_a} 行)")
print(f"    原因: 评分为核心字段，缺失则行无效")

# 策略B: 用均值填充 year
year_mean = df.select(avg(col("year").cast(FloatType()))).first()[0]
cleaned = cleaned.fillna({"year": year_mean})
print(f"\n3b. 用均值({year_mean:.1f})填充 year 缺失值")
print(f"    原因: 年份缺失可用均值填充，保留该行其他信息")

print(f"\n清洗后: {cleaned.count()} 行")

print("\n4. 基本统计:")
cleaned.select("rating_score", "rating_count", "year", "collect_count").describe().show()

# ============================
# A-2: Spark SQL 统计分析
# ============================
print("\n" + "=" * 60)
print("A-2: Spark SQL 统计分析")
print("=" * 60)

cleaned.createOrReplaceTempView("movies")

# 查询1: GROUP BY 聚合
print("\n【查询1】GROUP BY: 各年份电影数量与评分 (Top 10)")
spark.sql("""
    SELECT CAST(year AS INT) AS year, COUNT(*) AS cnt,
           ROUND(AVG(rating_score), 2) AS avg_score
    FROM movies WHERE year > 1980
    GROUP BY year ORDER BY year DESC LIMIT 10
""").show(10, truncate=False)

# 查询2: Top-N
print("\n【查询2】Top-N: 评分最高的 10 部电影 (评价>100000)")
spark.sql("""
    SELECT title, rating_score, rating_count, CAST(year AS INT) AS year, genres
    FROM movies WHERE rating_count > 100000
    ORDER BY rating_score DESC, rating_count DESC LIMIT 10
""").show(10, truncate=False)

# 查询3: 时间趋势
print("\n【查询3】时间趋势: 按年代统计")
spark.sql("""
    SELECT CONCAT(FLOOR(year/10)*10, 's') AS decade,
           COUNT(*) AS cnt, ROUND(AVG(rating_score), 2) AS avg_score,
           ROUND(AVG(rating_count), 0) AS avg_votes
    FROM movies WHERE year >= 1950 AND year <= 2020
    GROUP BY FLOOR(year/10)*10 ORDER BY decade
""").show(10, truncate=False)

# 查询4: 窗口函数
print("\n【查询4】窗口函数: 各年代评分最高类型 Top 3")
from pyspark.sql.functions import split as spark_split
df_genre = cleaned.withColumn("main_genre", spark_split(col("genres"), "/")[0]) \
    .withColumn("decade", (col("year") / 10).cast("int") * 10)
df_genre.createOrReplaceTempView("movies_genre")
spark.sql("""
    SELECT decade, main_genre, genre_avg_score, rnk FROM (
        SELECT decade, main_genre,
               ROUND(AVG(rating_score), 2) AS genre_avg_score,
               ROW_NUMBER() OVER (PARTITION BY decade ORDER BY AVG(rating_score) DESC) AS rnk
        FROM movies_genre
        WHERE decade >= 1980 AND main_genre IS NOT NULL AND main_genre != ''
        GROUP BY decade, main_genre HAVING COUNT(*) >= 5
    ) WHERE rnk <= 3 ORDER BY decade DESC, rnk
""").show(30, truncate=False)

print("\n全部分析完成!")
spark.stop()
