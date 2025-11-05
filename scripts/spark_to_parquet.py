## READ THROUGH THIS AGAIN

import os, sys, findspark
from pyspark.sql import SparkSession

os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable 
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

findspark.init()

# --- Spark session (reuse yours if already created) ---
spark = (
    SparkSession.builder
    .appName("SpotifyRec")
    .master("local[4]")
    .config("spark.driver.memory", "10g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4")
    .getOrCreate()
)

PATH = "../parquet_data/"
os.makedirs(PATH, exist_ok=True)
JDBC_URL = "jdbc:postgresql://localhost:5432/mpd?sslmode=disable"

try:
    for t in ["playlists", "tracks"]:
        spark.read.format("jdbc") \
            .option("url", JDBC_URL) \
            .option("dbtable", f"train.{t}") \
            .option("user", "admin") \
            .option("password", "admin") \
            .option("driver", "org.postgresql.Driver") \
            .load() \
            .write.mode("overwrite").parquet(f"{PATH}{t}")
except Exception as e:
    print("Error reading/writing playlists or tracks:", e)


bounds = (
    spark.read.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "(select min(pid) lo, max(pid) hi from train.playlist_tracks) b")
        .option("user", "admin")
        .option("password", "admin")
        .option("driver", "org.postgresql.Driver")
        .load()
        .first()
)
lo, hi = int(bounds.lo), int(bounds.hi)
print(f"pid range: {lo} → {hi}")

# Partitioned Read + Write
(
    spark.read.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "train.playlist_tracks")
        .option("user", "admin")
        .option("password", "admin")
        .option("driver", "org.postgresql.Driver")
        .option("partitionColumn", "pid")
        .option("lowerBound", lo)
        .option("upperBound", hi)
        .option("numPartitions", 16)      # tune 8–32 based on cores/RAM
        .option("fetchsize", 10000)
        .load()
        .select("pid", "pos", "track_uri")
        .repartition(16)
        .write
        .option("maxRecordsPerFile", 2_000_000)
        .mode("overwrite")
        .parquet(f"{PATH}playlist_tracks")
)

print("Done. Check for _SUCCESS files in {PATH}")