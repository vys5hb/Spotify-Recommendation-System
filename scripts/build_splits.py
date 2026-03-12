#!/usr/bin/env python3
# Import Libraries
import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

# Create the Spark session that does the heavy dataframe work for this script.
def create_spark_session(app_name, master, driver_memory):
    return (
        SparkSession.builder.appName(app_name) # name of Spark app (spotify-mpd-ingestion)
        .master(master) # where the Spark Session will run (local)
        .config("spark.driver.memory", driver_memory) # amount of RAM to give process (4 - 8GB)
        .config("spark.sql.shuffle.partitions", "64") # when dealing with groupBy, joins, aggregations, we split into 64 partitions instead of the default 200 (easier on local device)
        .config("spark.sql.session.timeZone", "UTC") # sets timezone to UTC
        .getOrCreate() # if SparkSession already exists, return it, otherwise create a new one
    )


# Reads parquet tables from data/silver
def read_silver_tables(input_dir, spark):
    input_root = Path(input_dir)
    playlists_path = input_root / "playlists.parquet"
    playlist_tracks_path = input_root / "playlist_tracks.parquet"

    if not playlists_path.exists():
        raise FileNotFoundError(f"Missing playlists table: {playlists_path}")
    if not playlist_tracks_path.exists():
        raise FileNotFoundError(f"Missing playlist_tracks table: {playlist_tracks_path}")

    playlists_df = spark.read.parquet(str(playlists_path))
    playlist_tracks_df = spark.read.parquet(str(playlist_tracks_path))
    return playlists_df, playlist_tracks_df


# Returns a DataFrame with pid, and accurate playlist_length
def build_playlist_lengths(playlists_df, playlist_tracks_df):
    # Count the tracks in each playlist
    track_counts = (
        playlist_tracks_df.groupBy("pid")
        .agg(F.count(F.lit(1)).cast("int").alias("playlist_length_from_tracks"))
    )

    return (
        playlists_df.select(
            F.col("pid").cast("long").alias("pid"),
            F.col("num_tracks").cast("int").alias("num_tracks"),
        )
        .join(track_counts, on="pid", how="left")
        .select(
            F.col("pid"),
            F.coalesce(
                F.col("playlist_length_from_tracks"), # chooses our created playlist_length_from_tracks first, if both are null, then set to 0
                F.col("num_tracks"),
                F.lit(0),
            ).cast("int").alias("playlist_length"),
        )
    )


# Deterministically assign each playlist to either train, validation, or test
# Adds a categorized split column to a DataFrame
# Returns a DataFrame with pid, split, playlist_length, eligible_for_eval
def assign_playlist_splits(playlist_lengths_df, train_ratio, validation_ratio, min_playlist_length, seed):
    # Deterministic hash number used to determine split
    split_score = (
        F.pmod(
            F.xxhash64(
                F.lit(str(seed)), 
                F.col("pid").cast("string")),
            F.lit(1000000),
        )
        / F.lit(1000000.0)
    )

    return (
        playlist_lengths_df
        .withColumn("split_score", split_score)
        .withColumn("split",
            F.when(F.col("playlist_length") < F.lit(min_playlist_length), F.lit("train")) # If length < min length, then put to train
            .when(F.col("split_score") < F.lit(train_ratio), F.lit("train")) # If score < train cutoff score, then put to train
            .when(F.col("split_score") < F.lit(train_ratio + validation_ratio), F.lit("validation")) # If score < train + validation cutoff score, then put to validation
            .otherwise(F.lit("test"))) # All others put to test
        .withColumn("eligible_for_eval", F.col("split").isin("validation", "test"))
        .select("pid", "split", "playlist_length", "eligible_for_eval")
    )


# Builds training split of playlist_tracks DataFrame 
def build_train_tracks(playlist_tracks_df, playlist_splits_df):
    train_pids = playlist_splits_df.where(F.col("split") == "train").select("pid") # A DataFrame of 'pid' where "split" == "train"
    # Essentially finds all pids which are assigned to train
    return (
        playlist_tracks_df
        .join(train_pids, on="pid", how="inner")
        .select('pid', 'pos', 'track_id', 'artist_id', 'album_id', 'duration_ms')
    ) 


# main() calls this function twice to mask both our validation & test splits.
# Creates validation/test splits, also builds context/target masks
def build_masked_split(playlist_tracks_df, playlist_splits_df, split_name, seed, mask_fraction, max_hidden):
    # Creates a value for the max number of target tracks each pid can have. This number caps at 10 for every pid.
    # mask_fraction = .20, max_hidden = 10. 
    # If a playlist has 50+ songs it will reach the max_hidden. 
    eval_playlists = (
        playlist_splits_df.where(F.col("split") == split_name)
        .select("pid", "playlist_length")
        .withColumn(
            "hidden_count",
            F.least(
                F.lit(max_hidden),
                F.greatest(
                    F.lit(1),
                    F.ceil(F.col("playlist_length") * F.lit(mask_fraction)).cast("int"),
                ),
            ),
        )
    )

    # Creates mask_score, a deterministic hash between 0 <= x < 1000000. Hash is built from seed, split_name, pid, and pos.
    masked_rows = (
        playlist_tracks_df.join(eval_playlists, on="pid", how="inner")
        .withColumn(
            "mask_score",
            F.pmod(
                F.xxhash64(
                    F.lit(str(seed)),
                    F.lit(split_name),
                    F.col("pid").cast("string"),
                    F.col("pos").cast("string"),
                ),
                F.lit(1000000.0),
            ),
        )
    )

    # Creates a random deterministic row ranking within each pid, mask_rank
    rank_window = Window.partitionBy("pid").orderBy(F.col("mask_score"), F.col("pos"))
    ranked_rows = masked_rows.withColumn("mask_rank", F.row_number().over(rank_window))

    # max_hidden = 10
    # mask_rank is used to pick up to 10 targets inside each pid. context_df takes the rest of the songs after we hit our hidden_count
    context_df = ranked_rows.where(F.col("mask_rank") > F.col("hidden_count")).select('pid', 'pos', 'track_id', 'artist_id', 'album_id', 'duration_ms')
    targets_df = ranked_rows.where(F.col("mask_rank") <= F.col("hidden_count")).select('pid', 'pos', 'track_id', 'artist_id', 'album_id', 'duration_ms')
    return context_df, targets_df


# Checks that the split ratios add up to 1.0
def validate_ratios(train_ratio, validation_ratio, test_ratio):
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must add up to 1.0, got {total_ratio}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build train, validation, and test splits from silver Parquet tables.")
    parser.add_argument("--input", required=True, help="Directory containing the silver Parquet tables.")
    parser.add_argument("--output", required=True, help="Directory where the gold split tables will be written.")
    parser.add_argument("--master", default="local[*]", help="Spark master URL. Default: local[*].")
    parser.add_argument("--app-name", default="spotify-mpd-splits", help="Spark application name.")
    parser.add_argument("--driver-memory", default="4g", help="Spark driver memory. Example: 4g or 8g.")
    parser.add_argument("--train-ratio", type=float, default=0.90, help="Train split ratio. Default: 0.90.")
    parser.add_argument("--validation-ratio", type=float, default=0.05, help="Validation split ratio. Default: 0.05.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio. Default: 0.05.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic split and mask hashing.")
    parser.add_argument("--min-playlist-length", type=int, default=5, help="Minimum playlist length for masked evaluation.")
    parser.add_argument("--mask-fraction", type=float, default=0.20, help="Fraction of each eval playlist to hide.")
    parser.add_argument("--max-hidden", type=int, default=10, help="Maximum number of hidden songs per eval playlist.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet outputs.")
    return parser.parse_args()


# Uses the helper functions to split playlists, build masked eval targets, and write the gold parquet tables.
# Reads in data/silver tables, makes train/val/test split, build masked context/target DataFrames, writes final parquet tables to data/gold
def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio) # Ensures that the ratios add up to 1

    output_root = Path(args.output) # Path object to data/gold
    output_root.mkdir(parents=True, exist_ok=True) # Creates data/gold if doesn't exist

    # Creates path for each parquet table write
    playlist_splits_path = output_root / "playlist_splits.parquet"
    train_playlist_tracks_path = output_root / "train_playlist_tracks.parquet"
    validation_context_path = output_root / "validation_context.parquet"
    validation_targets_path = output_root / "validation_targets.parquet"
    test_context_path = output_root / "test_context.parquet"
    test_targets_path = output_root / "test_targets.parquet"
    write_mode = "overwrite" if args.overwrite else "errorifexists"

    # Creates SparkSession
    spark = create_spark_session(args.app_name, args.master, args.driver_memory)

    try:
        playlists_df, playlist_tracks_df = read_silver_tables(args.input, spark) # Reads parquet tables from data/silver into Spark DataFrames
        playlist_lengths_df = build_playlist_lengths(playlists_df, playlist_tracks_df) # Builds a DataFrame with pid, and accurate playlist length
        playlist_splits_df = assign_playlist_splits( # Builds DataFrame with pid, split, playlist length, eligible for eval
            playlist_lengths_df,
            args.train_ratio,
            args.validation_ratio,
            args.min_playlist_length,
            args.seed,
        )

        train_playlist_tracks_df = build_train_tracks(playlist_tracks_df, playlist_splits_df)
        # Builds our validation masked DataFrames with default settings from args
        validation_context_df, validation_targets_df = build_masked_split(
            playlist_tracks_df,
            playlist_splits_df,
            "validation",
            args.seed,
            args.mask_fraction,
            args.max_hidden,
        )
        # Builds our test masked DataFrames with default settings from args
        test_context_df, test_targets_df = build_masked_split(
            playlist_tracks_df,
            playlist_splits_df,
            "test",
            args.seed,
            args.mask_fraction,
            args.max_hidden,
        )

        # Writes train/val/test tables to parquet in data/gold
        playlist_splits_df.write.mode(write_mode).parquet(str(playlist_splits_path))
        train_playlist_tracks_df.write.mode(write_mode).parquet(str(train_playlist_tracks_path))
        validation_context_df.write.mode(write_mode).parquet(str(validation_context_path))
        validation_targets_df.write.mode(write_mode).parquet(str(validation_targets_path))
        test_context_df.write.mode(write_mode).parquet(str(test_context_path))
        test_targets_df.write.mode(write_mode).parquet(str(test_targets_path))

        # Validation summary to ensure data and paths are correct
        written_splits_df = spark.read.parquet(str(playlist_splits_path))
        split_counts = {
            row["split"]: row["count"]
            for row in written_splits_df.groupBy("split").count().collect()
        }

        summary = {
            "input_path": str(Path(args.input)),
            "output_path": str(output_root),
            "split_counts": split_counts,
            "eligible_eval_playlists": split_counts.get("validation", 0) + split_counts.get("test", 0),
            "playlist_splits_path": str(playlist_splits_path),
            "train_playlist_tracks_path": str(train_playlist_tracks_path),
            "validation_context_path": str(validation_context_path),
            "validation_targets_path": str(validation_targets_path),
            "test_context_path": str(test_context_path),
            "test_targets_path": str(test_targets_path),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        spark.stop() # Closes spark connection


if __name__ == "__main__":
    main()
