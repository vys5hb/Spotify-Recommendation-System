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

def insert_tracks(df_tracks: pd.DataFrame):
    if df_tracks.empty: return
    df = df_tracks.drop_duplicates(subset=["track_uri"]).copy()
    with ENGINE.begin() as conn:
        conn.execute(text("""
          CREATE TEMP TABLE tmp_tracks(
            track_uri TEXT, 
            track_name TEXT, 
            artist_uri TEXT, 
            artist_name TEXT, 
            album_uri TEXT, 
            album_name TEXT, 
            track_duration BIGINT
          ) ON COMMIT DROP;
        """))
        df.to_sql("tmp_tracks", conn, if_exists="append", index=False)
        
        conn.execute(text("""
          INSERT INTO train.tracks(track_uri, track_name, artist_uri, artist_name, album_uri, album_name, track_duration)
          SELECT track_uri, track_name, artist_uri, artist_name, album_uri, album_name, track_duration
          FROM tmp_tracks
          ON CONFLICT (track_uri) DO UPDATE
          SET track_name     = EXCLUDED.track_name,
              artist_uri     = EXCLUDED.artist_uri,
              artist_name    = EXCLUDED.artist_name,
              album_uri      = EXCLUDED.album_uri,
              album_name     = EXCLUDED.album_name,
              track_duration = COALESCE(train.tracks.track_duration, EXCLUDED.track_duration);  
        """))

def insert_playlists(df_playlists: pd.DataFrame):
    if df_playlists.empty: return
    df = df_playlists.drop_duplicates(subset=["pid"]).copy()
    with ENGINE.begin() as conn:
        conn.execute(text("""
          CREATE TEMP TABLE tmp_playlists(
            pid BIGINT,
            name TEXT,
            num_tracks INT,
            num_artists INT,
            num_albums INT,
            num_followers INT,
            num_edits INT,
            playlist_duration BIGINT,
            modified_at BIGINT,
            collaborative BOOLEAN
          ) ON COMMIT DROP;           
        """))
        df.to_sql("tmp_playlists", conn, if_exists="append", index=False)

        conn.execute(text("""
          INSERT INTO train.playlists(pid, name, num_tracks, num_artists, num_albums, num_followers, num_edits, playlist_duration, modified_at, collaborative)
          SELECT pid, name, num_tracks, num_artists, num_albums, num_followers, num_edits, playlist_duration, modified_at, collaborative
          FROM tmp_playlists
          ON CONFLICT (pid) DO UPDATE
          SET name              = EXCLUDED.name,
              num_tracks        = EXCLUDED.num_tracks,
              num_artists       = EXCLUDED.num_artists,
              num_albums        = EXCLUDED.num_albums,
              num_followers     = EXCLUDED.num_followers,
              num_edits         = EXCLUDED.num_edits,
              playlist_duration = EXCLUDED.playlist_duration,
              modified_at       = EXCLUDED.modified_at,
              collaborative     = EXCLUDED.collaborative;  
        """))

def insert_playlist_tracks(df_edges: pd.DataFrame):
    if df_edges.empty: return
    with ENGINE.begin() as conn:
        conn.execute(text("""
          CREATE TEMP TABLE tmp_edges(
              pid BIGINT,
              pos INT,
              track_uri TEXT
          )   ON COMMIT DROP;
        """))
        df_edges.to_sql("tmp_edges", conn, if_exists="append", index=False)
        
        conn.execute(text("""
          INSERT INTO train.playlist_tracks(pid, pos, track_uri)
          SELECT pid, pos, track_uri
          FROM tmp_edges
          ON CONFLICT (pid, pos) DO NOTHING;
        """))

def main():
    slice_paths = sorted(glob.glob(SLICE_GLOB))
    print(f"Found {len(slice_paths)} slice files")
    
    for path in tqdm(slice_paths, desc="Loading MPD slices"):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)

        pls, edges, trks = [], [], []
        for pl in blob["playlists"]:
            pls.append({
                "pid": int(pl["pid"]),
                "name": pl.get("name"),
                "num_tracks": int(pl.get("num_tracks", 0)),
                "num_artists": int(pl.get("num_artists", 0)),
                "num_albums": int(pl.get("num_albums", 0)),
                "num_followers": int(pl.get("num_followers", 0)),
                "num_edits": int(pl.get("num_edits", 0)),
                "playlist_duration": int(pl.get("duration_ms", 0)),
                "modified_at": int(pl.get("modified_at", 0)),
                "collaborative": (pl.get("collaborative", "false") == "true")
            })
            for t in pl["tracks"]:
                trks.append({
                    "track_uri": t["track_uri"],
                    "track_name": t.get("track_name"),
                    "artist_uri": t.get("artist_uri"),
                    "artist_name": t.get("artist_name"),
                    "album_uri": t.get("album_uri"),
                    "album_name": t.get("album_name"),
                    "track_duration": int(t.get("duration_ms", 0))
                })
                edges.append({
                    "pid": int(pl["pid"]), 
                    "pos": int(t.get("pos", 0)), 
                    "track_uri": t["track_uri"]})

        insert_tracks(pd.DataFrame(trks))
        insert_playlists(pd.DataFrame(pls))
        insert_playlist_tracks(pd.DataFrame(edges))

if __name__ == "__main__":
    main()
