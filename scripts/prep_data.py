# Import Libraries
import os, sys, findspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, Window as W


# Uses Java 17 & Python 3.11
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
# Builds PySpark session for 4 local cores with 10GB RAM
findspark.init()
spark = (
    SparkSession.builder
    .appName("SpotifyRec")
    .master("local[4]")
    .config("spark.driver.memory", "10g")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
# Remove error logs for cleaner output
spark.sparkContext.setLogLevel("ERROR")

playlist_tracks = spark.read.parquet("parquet_data/playlist_tracks")
playlists = spark.read.parquet("parquet_data/playlists")
tracks = spark.read.parquet("parquet_data/tracks")

# Creates a unique tid (int) for each track_uri (string).
# Item vocabulary
track_id = (
    tracks.select("track_uri")
    .dropDuplicates()
    .withColumn("tid", F.dense_rank().over(W.orderBy("track_uri")) - 1)
)

# Saves the track metadata for later training and inference.
# Item features
track_features = (
    tracks.join(track_id, on="track_uri", how="inner")
    .select('tid', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'track_duration')
)
# Write track_features to parquet for model training
track_features.write.mode('overwrite').parquet('parquet_data/track_features')

# Saves the playlist metadata for later training and inference.
# User Features
playlist_features = (
    playlists.select('pid', 'name', 'num_tracks', 'num_albums', 'num_artists', 'num_followers', 'collaborative', 'modified_at')
)
# Write playlist_features to parquet for model training
playlist_features.write.mode('overwrite').parquet('parquet_data/playlist_features')

# Joins unique tid(int) to each pid(int) from playlist_tracks for collaborative filtering.
# Edges are the interaction matrix
edges = (
    playlist_tracks.join(track_id, on='track_uri', how='inner')
    .select('pid', 'tid')
)
edges.write.mode('overwrite').parquet('parquet_data/edges')