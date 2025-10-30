CREATE SCHEMA IF NOT EXISTS train;

CREATE TABLE IF NOT EXISTS train.tracks (
  track_uri      TEXT PRIMARY KEY,
  track_name     TEXT,
  artist_uri     TEXT,
  artist_name    TEXT,
  album_uri      TEXT,
  album_name     TEXT,
  track_duration BIGINT
);

CREATE TABLE IF NOT EXISTS train.playlists (
  pid               BIGINT PRIMARY KEY,
  name              TEXT,
  num_tracks        INT,
  num_artists       INT,
  num_albums        INT,
  num_followers     INT,
  num_edits         INT,
  playlist_duration BIGINT,
  modified_at       BIGINT,
  collaborative     BOOLEAN
);

CREATE TABLE IF NOT EXISTS train.playlist_tracks (
  pid        BIGINT,
  pos        INT,
  track_uri  TEXT,
  PRIMARY KEY (pid, pos),
  FOREIGN KEY (pid) REFERENCES train.playlists(pid),
  FOREIGN KEY (track_uri) REFERENCES train.tracks(track_uri)
);

CREATE INDEX IF NOT EXISTS idx_pl_tracks_pid   ON train.playlist_tracks(pid);
CREATE INDEX IF NOT EXISTS idx_pl_tracks_track ON train.playlist_tracks(track_uri);
