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
# Item vocabulary - Track
track_id = (
    tracks.select("track_uri")
    .dropDuplicates()
    .withColumn("tid", F.dense_rank().over(W.orderBy("track_uri")) - 1)
)
# Calculate # of playlists each track appears in
track_cnt = (
    playlist_tracks.join(track_id, on='track_uri', how='inner')
    .groupBy('tid')
    .agg(F.countDistinct('pid').alias('track_cnt'))
)
# Saves the track metadata for later training and inference.
# Item features - Track
track_features = (
    tracks.join(track_id, on="track_uri", how="inner")
    .join(track_cnt, on='tid', how='left')
    .select('tid', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'track_duration', 'track_cnt')
)
# Write track_features to parquet for model training
track_features.write.mode('overwrite').parquet('parquet_data/track_features')

# Creates a unique aid (int) for each artist_uri (string).
# Item vocabulary - Artist
artist_id = (
    tracks.select('artist_uri', 'artist_name')
    .dropDuplicates(['artist_uri'])
    .withColumn("aid", F.dense_rank().over(W.orderBy("artist_uri")) - 1)
)
# Calculate # of playlists each artist appears in, and total # of appearances for each artist
artist_cnt = (
    playlist_tracks.alias('pt')
    .join(tracks.select('track_uri', 'artist_uri'), on='track_uri', how='inner')
    .groupBy('artist_uri')
    .agg(F.countDistinct('pt.pid').alias('artist_playlist_cnt'),
         F.count('pt.pid').alias('artist_cnt')
    )
)
# Item features - Artist
artist_features = (
    artist_id.join(artist_cnt, on='artist_uri', how='left')
    .select('aid', 'artist_uri', 'artist_name', 'artist_playlist_cnt', 'artist_cnt')
)
# Write artist_features to parquet for model training
artist_features.write.mode('overwrite').parquet('parquet_data/artist_features')

# Saves the playlist metadata for later training and inference.
# User Features
latest_edit = playlists.agg(F.max("modified_at").alias("now")).collect()[0]["now"]
playlist_features = (
    playlists.select('pid', 'name', 'num_tracks', 'num_artists', 'num_albums', 'playlist_duration', 'modified_at', 'collaborative')
    .withColumn("days_since_modified", (F.lit(latest_edit) - F.col("modified_at")) / 86400)
    .drop("modified_at")
    .select('pid', 'name', 'num_tracks', 'num_artists', 'num_albums', 'playlist_duration', 'days_since_modified', 'collaborative')
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