CREATE SCHEMA IF NOT EXISTS mpd;

CREATE TABLE IF NOT EXISTS mpd.tracks (
  track_uri   TEXT PRIMARY KEY,
  track_name  TEXT,
  artist_name TEXT,
  album_name  TEXT
);

CREATE TABLE IF NOT EXISTS mpd.playlists (
  pid           BIGINT PRIMARY KEY,
  name          TEXT,
  num_tracks    INT,
  num_followers INT,
  collaborative BOOLEAN,
  modified_at   BIGINT
);

CREATE TABLE IF NOT EXISTS mpd.playlist_tracks (
  pid        BIGINT,
  pos        INT,
  track_uri  TEXT,
  PRIMARY KEY (pid, pos),
  FOREIGN KEY (pid) REFERENCES mpd.playlists(pid),
  FOREIGN KEY (track_uri) REFERENCES mpd.tracks(track_uri)
);

CREATE INDEX IF NOT EXISTS idx_pl_tracks_pid   ON mpd.playlist_tracks(pid);
CREATE INDEX IF NOT EXISTS idx_pl_tracks_track ON mpd.playlist_tracks(track_uri);
