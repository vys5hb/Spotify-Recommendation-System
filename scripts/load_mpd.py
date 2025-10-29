import os, json, glob
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

PG_USER = os.getenv("POSTGRES_USER", "admin")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "admin")
PG_DB   = os.getenv("POSTGRES_DB", "mpd")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")

ENGINE = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@localhost:{PG_PORT}/{PG_DB}")

SLICE_GLOB = os.path.join(os.path.dirname(__file__), "..", "data", "mpd", "mpd.slice.*.json")

def upsert_tracks(df_tracks: pd.DataFrame):
    if df_tracks.empty: return
    df = df_tracks.drop_duplicates(subset=["track_uri"]).copy()
    with ENGINE.begin() as conn:
        conn.execute(text("""
          CREATE TEMP TABLE tmp_tracks(
            track_uri TEXT, track_name TEXT, artist_name TEXT, album_name TEXT
          ) ON COMMIT DROP;
        """))
        df.to_sql("tmp_tracks", conn, if_exists="append", index=False)
        conn.execute(text("""
          INSERT INTO mpd.tracks(track_uri, track_name, artist_name, album_name)
          SELECT track_uri, track_name, artist_name, album_name
          FROM tmp_tracks
          ON CONFLICT (track_uri) DO NOTHING;
        """))

def insert_playlists(df_playlists: pd.DataFrame):
    if df_playlists.empty: return
    with ENGINE.begin() as conn:
        df_playlists.drop_duplicates(subset=["pid"]).to_sql(
            "playlists", conn, schema="mpd", if_exists="append", index=False, method="multi"
        )

def insert_playlist_tracks(df_edges: pd.DataFrame):
    if df_edges.empty: return
    with ENGINE.begin() as conn:
        df_edges.to_sql("playlist_tracks", conn, schema="mpd", if_exists="append", index=False, method="multi")

def main():
    slice_paths = sorted(glob.glob(SLICE_GLOB))
    print(f"Found {len(slice_paths)} slice files")
    for path in tqdm(slice_paths, desc="Loading MPD slices"):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)

        pls, edges, trks = [], [], []
        for pl in blob["playlists"]:
            pls.append({
                "pid": pl["pid"],
                "name": pl.get("name"),
                "num_tracks": pl.get("num_tracks"),
                "num_followers": pl.get("num_followers"),
                "collaborative": (pl.get("collaborative", "false") == "true"),
                "modified_at": pl.get("modified_at"),
            })
            for t in pl["tracks"]:
                trks.append({
                    "track_uri": t["track_uri"],
                    "track_name": t.get("track_name"),
                    "artist_name": t.get("artist_name"),
                    "album_name": t.get("album_name"),
                })
                edges.append({"pid": pl["pid"], "pos": t["pos"], "track_uri": t["track_uri"]})

        upsert_tracks(pd.DataFrame(trks))
        try:
            insert_playlists(pd.DataFrame(pls))
        except Exception:
            pass
        try:
            insert_playlist_tracks(pd.DataFrame(edges))
        except Exception:
            pass

if __name__ == "__main__":
    main()
