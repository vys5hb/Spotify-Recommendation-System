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

edges_shift = edges.withColumn("tid", F.col("tid") + 1)
track_features_shift = track_features.withColumn("tid", F.col("tid") + 1)

# Building the length = 20, arrays for the tids, and create a masking object
PAD_ID = 0
max_length = 20


# For each playlist, create an array of all track ids in that playlist
pl_seqs = (edges_shift
         .groupby('pid')
         .agg(F.array_distinct(F.collect_list('tid')).alias('tids'))
         .filter(F.size('tids') >= 2)
)

pairs = (
    pl_seqs
    # 1) Pick a positive by shuffling the tids per playlist, then take the first
    .withColumn('shuffle', F.shuffle(F.col('tids')))
    .withColumn('pos_tid', F.element_at(F.col('shuffle'), 1))
    # 2) Remove the positive from the context pool
    .withColumn('remain', F.filter('tids', lambda x: x != F.col('pos_tid')))
    # 3) Take up the max_length random items, shuffle then slice
    .withColumn('items', F.slice(F.shuffle(F.col('remain')), 1, max_length))
    # 4) Build mask & padding to max_length
    .withColumn('len', F.size('items'))
    .withColumn('pad_len', F.greatest(F.lit(0), F.lit(max_length) - F.col('len')))
    .withColumn('tokens', F.concat(F.col('items'), F.array_repeat(F.lit(PAD_ID), F.col('pad_len'))))
    .withColumn('mask', F.concat(F.array_repeat(F.lit(1), F.col('len')), F.array_repeat(F.lit(0), F.col('pad_len'))))
    # 5) Select relevant items
    .select('pid', 'tokens', 'mask', 'pos_tid')
)
pairs.show(15, truncate=False)

# Creates a random 95/5 train/validation split
bucketed = pairs.withColumn('bucket', F.pmod(F.abs(F.hash('pid')), F.lit(100))) # Creates a random positive integer & modulus divides by 100. Essentially randomly groups each pid into 100 buckets
train_pairs = bucketed.filter('bucket < 95').select('pid', 'tokens', 'mask', 'pos_tid') # ~95% of the data
val_pairs = bucketed.filter('bucket >= 95').select('pid', 'tokens', 'mask', 'pos_tid') # ~5% of the data

print("train rows:", train_pairs.count())
print("val rows:",   val_pairs.count())

n_tracks = edges_shift.agg(F.max('tid').alias('max_tid')).collect()[0]['max_tid']
print('n_tracks (embedding size):', int(n_tracks) + 1)

train_pairs.write.mode('overwrite').parquet('parquet_data/train_pairs')
val_pairs.write.mode('overwrite').parquet('parquet_data/val_pairs')