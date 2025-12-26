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

# Create a unique track id, artist id, and album id
track_id = (
    tracks.select('track_uri', 'artist_uri', 'album_uri')
    .dropDuplicates()
    .withColumn("tid", F.dense_rank().over(W.orderBy("track_uri")))
    .withColumn('aid', F.dense_rank().over(W.orderBy('artist_uri')))
    .withColumn('alid', F.dense_rank().over(W.orderBy('album_uri')))
)

# Joins unique tid(int) to each pid(int) from playlist_tracks for collaborative filtering.
# Edges are the interaction matrix
edges = (
    playlist_tracks.join(track_id, on='track_uri', how='inner')
    .select('pid', 'tid')
)
edges.write.mode('overwrite').parquet('parquet_data/edges')
print('Successfully saved edges.')
# Building the length = 20, arrays for the tids, and create a masking object
PAD_ID = 0
max_length = 20

# For each playlist, create an array of all track ids in that playlist
pl_seqs = (edges
         .groupby('pid')
         .agg(F.array_distinct(F.collect_list('tid')).alias('tids'))
         .filter(F.size('tids') >= 2)
)
# Creates unique tokens per token id, and a masking array for filler tokens. Randomly selects a positive token id to train model on positive associations
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

# Creates a random 95/5 train/validation split
bucketed = pairs.withColumn('bucket', F.pmod(F.abs(F.hash('pid')), F.lit(100))) # Creates a random positive integer & modulus divides by 100. Essentially randomly groups each pid into 100 buckets
train_pairs = bucketed.filter('bucket < 95').select('pid', 'tokens', 'mask', 'pos_tid') # ~95% of the data
val_pairs = bucketed.filter('bucket >= 95').select('pid', 'tokens', 'mask', 'pos_tid') # ~5% of the data

# Write train_pairs to parquet for model training
train_pairs.coalesce(16).write.mode('overwrite').parquet('parquet_data/train_pairs')
print('Successfully saved train_pairs.')
# Write val_pairs to parquet for model training
val_pairs.write.mode('overwrite').parquet('parquet_data/val_pairs')
print('Successfully saved val_pairs.')

# Select only the playlist ids from the train data split
train_edges = edges.join(train_pairs, on='pid', how='inner')

# Calculate # of playlists each track appears in only from the train data split
train_track_cnt = train_edges.groupBy('tid').agg(F.countDistinct('pid').alias('track_cnt'))

# Calculate # of playlists each artist appears in only from the train data split
train_artist_cnt = (
    train_edges.join(track_id.select('tid', 'artist_uri'), on='tid', how='inner')
    .groupBy('artist_uri')
    .agg(F.countDistinct('pid').alias('artist_cnt'))
)
# Saves the track metadata for later training and inference.
# Item features
track_features = (
    track_id
    .join(train_track_cnt, on="tid", how="left")
    .join(train_artist_cnt, on='artist_uri', how='left')
    .select('tid', 'aid', 'alid', 'track_cnt', 'artist_cnt')
    .fillna({'track_cnt': 0, 'artist_cnt': 0})
)
track_features = (
    track_features
    .withColumn('log_track_cnt', F.log1p('track_cnt'))
    .withColumn('log_artist_cnt', F.log1p('artist_cnt'))
)
z_score = track_features.agg(
    F.mean('log_track_cnt').alias('mean_track_cnt'),
    F.stddev_pop('log_track_cnt').alias('std_track_cnt'),
    F.mean('log_artist_cnt').alias('mean_artist_cnt'),
    F.stddev_pop('log_artist_cnt').alias('std_artist_cnt')
).collect()[0]
track_features = (
    track_features
    .withColumn('z_log_track_cnt', (F.col('log_track_cnt') - F.lit(z_score['mean_track_cnt'])) / F.lit(z_score['std_track_cnt']))
    .withColumn('z_log_artist_cnt', (F.col('log_artist_cnt') - F.lit(z_score['mean_artist_cnt'])) / F.lit(z_score['std_artist_cnt']))
    .select('tid', 'aid', 'alid', 'z_log_track_cnt', 'z_log_artist_cnt')
)
track_features.write.mode('overwrite').parquet('parquet_data/track_features')
print('Successfully saved track_features.')

# Saves the playlist metadata for later training and inference.
# User Features
latest_edit = playlists.agg(F.max("modified_at").alias("now")).collect()[0]["now"]
playlist_features = (
    playlists.select('pid', 'name', 'num_tracks', 'num_artists', 'num_albums', 'playlist_duration', 'modified_at', 'collaborative')
    .withColumn("days_since_modified", (F.lit(latest_edit) - F.col("modified_at")) / 86400)
    .withColumn('log_n_tracks', F.log1p('num_tracks'))
    .withColumn('log_n_artists', F.log1p('num_artists'))
    .withColumn('log_n_albums', F.log1p('num_albums'))
    .withColumn('log_pl_duration', F.log1p('playlist_duration'))
    .withColumn('log_days_mod', F.log1p('days_since_modified'))
    .select('pid', 'name', 'log_n_tracks', 'log_n_artists', 'log_n_albums', 'log_pl_duration', 'log_days_mod', 'collaborative')
)
# Write playlist_features to parquet for model training
playlist_features.write.mode('overwrite').parquet('parquet_data/playlist_features')
print('Successfully saved playlist_features.')