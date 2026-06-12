"""
豆瓣电影数据集分析 —— Spark SQL 统计分析
A-1: 数据清洗 (10分)
A-2: Spark SQL 统计分析 (15分)
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, stddev, min, max, mean, desc, asc, row_number, year, month
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType

spark = SparkSession.builder \
    .appName("DoubanMoviesAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# ============================================================
# A-1: 数据清洗
# ============================================================
print("=" * 60)
print("A-1: 数据清洗")
print("=" * 60)

# 1. 加载数据
df = spark.read.csv("file:///D:/大三下/云计算/douban_movies.csv", header=True, inferSchema=True, encoding="UTF-8")
print(f"\n1. 加载完成，原始数据行数: {df.count()}")
print("Schema:")
df.printSchema()
print("\n前 5 行:")
df.show(5, truncate=False)

# 2. 统计各字段缺失值比例
print("\n2. 各字段缺失值统计:")
total = df.count()
for c in df.columns:
    null_count = df.filter(col(c).isNull()).count()
    pct = null_count / total * 100
    print(f"  {c}: {null_count} 个缺失 ({pct:.2f}%)")

# 3. 两种不同的缺失值处理策略
# 策略 a: 删除 rating_score 为空的行（评分是核心字段，不能缺失）
cleaned_a = df.filter(col("rating_score").isNotNull())
rows_before_a = df.count()
rows_after_a = cleaned_a.count()
print(f"\n3a. 策略A - 删除 rating_score 缺失行: {rows_before_a} → {rows_after_a} (删除 {rows_before_a - rows_after_a} 行)")
print(f"    原因: 评分是核心分析字段，缺失会导致统计失真，故直接删除")

# 策略 b: 用均值填充 year 缺失值（年份可用均值替代，保留该行）
year_mean = df.select(avg(col("year").cast(FloatType()))).first()[0]
cleaned_b = cleaned_a.fillna({"year": year_mean})
year_nulls_before = cleaned_a.filter(col("year").isNull()).count()
print(f"\n3b. 策略B - 用均值({year_mean:.1f})填充 year 缺失: 填充了 {year_nulls_before} 个缺失值")
print(f"    原因: 年份缺失较少，用均值填充可保留该行其他字段信息")

# 最终清洗后
df_clean = cleaned_b
print(f"\n清洗前: {rows_before_a} 行")
print(f"清洗后: {df_clean.count()} 行")
print(f"共删除: {rows_before_a - df_clean.count()} 行")

# 4. 基本统计信息
print("\n4. 各字段基本统计:")
df_clean.select("rating_score", "rating_count", "year", "collect_count") \
    .describe().show()

# ============================================================
# A-2: Spark SQL 统计分析 (4 个查询)
# ============================================================
print("\n" + "=" * 60)
print("A-2: Spark SQL 统计分析")
print("=" * 60)

df_clean.createOrReplaceTempView("movies")

# 查询1: GROUP BY 聚合 —— 各年份电影数量和平均评分
print("\n【查询1】GROUP BY 聚合: 各年份电影数量和平均评分 (Top 10)")
result1 = spark.sql("""
    SELECT
        CAST(year AS INT) AS year,
        COUNT(*) AS movie_count,
        ROUND(AVG(rating_score), 2) AS avg_rating
    FROM movies
    WHERE year IS NOT NULL AND year > 1980
    GROUP BY year
    ORDER BY year DESC
    LIMIT 10
""")
result1.show(10, truncate=False)
print("分析: 展示了近10年各年份电影数量及平均评分趋势，可以看出电影产量与评分的关系")

# 查询2: Top-N —— 评分最高的 10 部电影（按评分和评价人数排序）
print("\n【查询2】ORDER BY Top-N: 高评分电影 Top 10 (评价人数 > 100000)")
result2 = spark.sql("""
    SELECT
        title,
        rating_score,
        rating_count,
        CAST(year AS INT) AS year,
        genres
    FROM movies
    WHERE rating_count > 100000
    ORDER BY rating_score DESC, rating_count DESC
    LIMIT 10
""")
result2.show(10, truncate=False)
print("分析: Top-N 查询筛出高评价且高评分的经典电影，反映了大众认可度最高的作品")

# 查询3: 时间维度趋势分析 —— 按年代统计电影数量和平均评分
print("\n【查询3】时间维度趋势: 按年代统计电影数量和评分变化")
result3 = spark.sql("""
    SELECT
        CONCAT(FLOOR(year/10)*10, 's') AS decade,
        COUNT(*) AS movie_count,
        ROUND(AVG(rating_score), 2) AS avg_rating,
        ROUND(AVG(rating_count), 0) AS avg_rating_count
    FROM movies
    WHERE year IS NOT NULL AND year >= 1950 AND year <= 2020
    GROUP BY FLOOR(year/10)*10
    ORDER BY decade
""")
result3.show(10, truncate=False)
print("分析: 按年代维度展示了电影产量和评分的变化趋势，可观察到电影产业发展的历史脉络")

# 查询4: 窗口函数 + JOIN —— 各类型电影排名（Top 3 类型）
print("\n【查询4】窗口函数: 各年代评分最高的电影类型")
# 拆分 genres 字段的第一个类型（主要类型）
from pyspark.sql.functions import split as spark_split

df_genre = df_clean.withColumn("main_genre", spark_split(col("genres"), "/")[0]) \
    .withColumn("decade", (col("year") / 10).cast("int") * 10)

df_genre.createOrReplaceTempView("movies_genre")

result4 = spark.sql("""
    SELECT decade, main_genre, genre_avg_rating, genre_rank
    FROM (
        SELECT
            decade,
            main_genre,
            ROUND(AVG(rating_score), 2) AS genre_avg_rating,
            ROW_NUMBER() OVER (PARTITION BY decade ORDER BY AVG(rating_score) DESC) AS genre_rank
        FROM movies_genre
        WHERE decade IS NOT NULL
            AND decade >= 1980
            AND main_genre IS NOT NULL
            AND main_genre != ''
        GROUP BY decade, main_genre
        HAVING COUNT(*) >= 5
    )
    WHERE genre_rank <= 3
    ORDER BY decade DESC, genre_rank
""")
result4.show(30, truncate=False)
print("分析: 用窗口函数 ROW_NUMBER() 按年代分组、按评分排名，可看出不同年代流行电影类型的变化")

print("\n" + "=" * 60)
print("全部分析完成!")
print("=" * 60)

spark.stop()
