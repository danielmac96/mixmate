import pandas as pd
import yt_dlp
import librosa
import numpy as np
import sqlite3
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import os


def make_song_id(row):
    base = f"{row['artist']}_{row['title']}_{row['track_id']}"
    return hashlib.md5(base.lower().encode()).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            song_id TEXT PRIMARY KEY,
            title TEXT,
            artist TEXT,
            artist_id TEXT,
            track_id TEXT,
            duration TEXT,
            duration_sec REAL,
            url TEXT,
            upload_date TEXT,
            likes INTEGER,
            reposts INTEGER,
            comments INTEGER,
            plays INTEGER,
            thumbnail TEXT,
            genre TEXT,
            filepath TEXT,
            downloaded INTEGER,
            source TEXT,
            bpm REAL,
            key TEXT,
            mode TEXT,
            camelot TEXT,
            beat_strength REAL,
            rhythm_regularity REAL,
            harmonic_clarity REAL,
            energy REAL,
            loudness REAL,
            bass_energy REAL,
            mid_energy REAL,
            high_energy REAL,
            brightness REAL,
            spectral_flatness REAL,
            spectral_contrast REAL,
            harmonic_energy REAL,
            percussive_energy REAL,
            percussiveness REAL,
            mfcc_1 REAL, mfcc_2 REAL, mfcc_3 REAL, mfcc_4 REAL, mfcc_5 REAL,
            mfcc_6 REAL, mfcc_7 REAL, mfcc_8 REAL, mfcc_9 REAL, mfcc_10 REAL,
            mfcc_11 REAL, mfcc_12 REAL, mfcc_13 REAL,
            danceability REAL,
            acousticness REAL,
            instrumentalness REAL,
            vocal_presence REAL,
            valence REAL,
            energy_variance REAL,
            speechiness REAL,
            liveness REAL,
            warmth REAL,
            aggression REAL,
            complexity REAL,
            analyzed INTEGER,
            analyzed_at TEXT
        )
    """)
    conn.close()


def load_existing_ids():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT song_id FROM songs", conn)
    except:
        df = pd.DataFrame(columns=["song_id"])
    conn.close()
    return set(df["song_id"])


def save_to_db(df):
    df = df.copy()
    df["analyzed_at"] = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("songs", conn, if_exists="append", index=False)
    conn.close()


def get_playlist_track_info(playlist_url:str) -> pd.DataFrame:

  ydl_opts = {'extract_flat': True, 'quiet': True}

  tracklist = []

  ydl = yt_dlp.YoutubeDL(ydl_opts)
  info = ydl.extract_info(playlist_url, download=False)
  playlist = info.get('entries', [])
  for track in playlist:
      tracklist.append(track['url'])

  ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}

  all_tracks = []

  for track_url in tracklist:

    track = ydl.extract_info(track_url, download=False)

    duration = track.get('duration', 0)
    mins = int(duration // 60) if duration else 0
    secs = int(duration % 60) if duration else 0
    duration_str = f"{mins}:{secs:02d}"

    all_tracks.append({
        'title': track.get('title'),
        'artist': track.get('uploader'),
        'artist_id': track.get('uploader_id'),
        'track_id': track.get('id'),
        'duration': duration_str,
        'duration_sec': track.get('duration'),
        'url': track.get('webpage_url'),
        'upload_date': track.get('upload_date'),
        'likes': track.get('like_count', 0),
        'reposts': track.get('repost_count', 0),
        'comments': track.get('comment_count', 0),
        'plays': track.get('view_count', 0),
        'thumbnail': track.get('thumbnail'),
        'genre': track.get('genre')
    })

  soundcloud_track_info_df = pd.DataFrame(all_tracks)

  return soundcloud_track_info_df


def download_playlist_files(soundcloud_track_info_df:pd.DataFrame, download_folder_path:str) -> pd.DataFrame:

  soundcloud_track_info_df['filepath'] = None
  soundcloud_track_info_df['downloaded'] = False
  soundcloud_track_info_df['source'] = None

  os.makedirs(download_folder_path, exist_ok=True)

  # Download + Analyze each track
  for idx, row in soundcloud_track_info_df.iterrows():
      filename = f"{row['artist']} - {row['title']}"
      filepath = os.path.join(download_folder_path, filename)

      # Download
      ydl_opts = {
          'format': 'bestaudio/best',
          'outtmpl': filepath,
          'postprocessors': [{
              'key': 'FFmpegExtractAudio',
              'preferredcodec': 'mp3',
              'preferredquality': '320',
          }],
          'quiet': True,
          'no_warnings': True,
          "nocheckcertificate": True,
          "http_headers": {
              "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
          },
          "ratelimit": 500_000,  # 500KB/s
          "concurrent_fragment_downloads": 1,
          "retries": 5,
          "fragment_retries": 5,
      }

      try:
          with yt_dlp.YoutubeDL(ydl_opts) as ydl:

              if row['duration_sec'] < 45:
                  print(f"⚠ SoundCloud preview detected for '{filename}' - searching YouTube...")

                  search_query = f"ytsearch1:{row['artist']} {row['title']} lyric audio"
                  with yt_dlp.YoutubeDL(ydl_opts) as ydl_yt:
                      yt_info = ydl_yt.extract_info(search_query, download=True)

                      if 'entries' in yt_info and len(yt_info['entries']) > 0:
                          video_info = yt_info['entries'][0]

                          duration_sec = video_info.get('duration', 0)
                          soundcloud_track_info_df.at[idx, 'duration_sec'] = duration_sec
                          soundcloud_track_info_df.at[idx, 'duration'] = f"{duration_sec // 60}:{duration_sec % 60:02d}"
                          soundcloud_track_info_df.at[idx, 'url'] = video_info.get('webpage_url', video_info.get('url', ''))

                  soundcloud_track_info_df.at[idx, 'source'] = 'YouTube'
                  print(f"✓ Downloaded from YouTube: {filename} ({soundcloud_track_info_df.at[idx, 'duration']})")
              else:
                  ydl.download([row['url']])
                  soundcloud_track_info_df.at[idx, 'source'] = 'SoundCloud'
                  print(f"✓ Downloaded from SoundCloud: {filename}")

          soundcloud_track_info_df.at[idx, 'filepath'] = filepath
          soundcloud_track_info_df.at[idx, 'downloaded'] = True

      except Exception as e:
          print(f"✗ Error downloading {filename}: {e}")
          try:
              search_query = f"ytsearch1:{row['artist']} {row['title']} lyric audio"
              with yt_dlp.YoutubeDL(ydl_opts) as ydl_yt:
                  yt_info = ydl_yt.extract_info(search_query, download=True)

                  if 'entries' in yt_info and len(yt_info['entries']) > 0:
                      video_info = yt_info['entries'][0]

                      duration_sec = video_info.get('duration', 0)
                      soundcloud_track_info_df.at[idx, 'duration_sec'] = duration_sec
                      soundcloud_track_info_df.at[idx, 'duration'] = f"{duration_sec // 60}:{duration_sec % 60:02d}"
                      soundcloud_track_info_df.at[idx, 'url'] = video_info.get('webpage_url', video_info.get('url', ''))

              soundcloud_track_info_df.at[idx, 'filepath'] = filepath
              soundcloud_track_info_df.at[idx, 'downloaded'] = True
              soundcloud_track_info_df.at[idx, 'source'] = 'YouTube'
              print(f"✓ Downloaded from YouTube: {filename} ({soundcloud_track_info_df.at[idx, 'duration']})")
          except Exception as e2:
              print(f"✗ Failed all sources for {filename}: {e2}")
              soundcloud_track_info_df.at[idx, 'downloaded'] = False
  return soundcloud_track_info_df

def analyze_downloaded_tracks(songs_df: pd.DataFrame, download_folder_path: str) -> pd.DataFrame:
    os.makedirs(download_folder_path, exist_ok=True)

    songs_df["analyzed"] = False

    print("Files in download folder:")
    for idx, row in songs_df.iterrows():
        filepath = row['filepath']

        if not filepath.endswith('.mp3'):
            filepath = filepath + ".mp3"

        print(f"Processing: {filepath}")

        try:
            y, sr = librosa.load(filepath)

            # === TEMPO & RHYTHM ===
            # BPM: Essential for beatmatching
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_value = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Beat strength: Strong beats = easier to beatmatch
            beat_strength = np.mean(onset_env[beats])

            # Rhythm regularity: Consistent rhythm = smoother transitions
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
            rhythm_regularity = 1 / (np.std(beat_intervals) + 0.001)

            # === KEY & HARMONIC MIXING ===
            # Key/Camelot: For harmonic mixing compatibility
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_idx = np.argmax(np.sum(chroma, axis=1))

            chroma_vals = np.sum(chroma, axis=1)
            is_major = chroma_vals[key_idx] > chroma_vals[(key_idx + 3) % 12]

            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = key_names[key_idx]
            mode = 'major' if is_major else 'minor'

            camelot_major = ['8B', '3B', '10B', '5B', '12B', '7B', '2B', '9B', '4B', '11B', '6B', '1B']
            camelot_minor = ['5A', '12A', '7A', '2A', '9A', '4A', '11A', '6A', '1A', '8A', '3A', '10A']
            camelot = camelot_major[key_idx] if is_major else camelot_minor[key_idx]

            # Harmonic clarity: How clear the key is (confident key detection = better harmonic mixing)
            harmonic_clarity = np.max(chroma_vals) / np.mean(chroma_vals)

            # === ENERGY & DYNAMICS ===
            # Energy: Overall track intensity for set flow
            energy = np.mean(librosa.feature.rms(y=y))

            # Loudness: For level matching between tracks
            loudness = np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y)))

            # === FREQUENCY SPECTRUM ===
            stft = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)

            # Bass/mid/high energy: For EQ-based mixing and energy matching
            bass_mask = (freqs >= 20) & (freqs < 250)
            mid_mask = (freqs >= 250) & (freqs < 4000)
            high_mask = (freqs >= 4000) & (freqs <= 20000)

            bass_energy = np.mean(stft[bass_mask, :])
            mid_energy = np.mean(stft[mid_mask, :])
            high_energy = np.mean(stft[high_mask, :])

            # Brightness: Normalized spectral centroid - bright vs dark sound
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            brightness = spectral_centroid / (sr / 2)

            # Spectral flatness: Tonal (musical) vs noisy - helps identify similar textures
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

            # === TIMBRE & TEXTURE ===
            # MFCC: Timbral fingerprint for finding similar-sounding tracks
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)

            # Spectral contrast: Peak/valley energy - helps identify genre/vibe
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

            # === HARMONIC vs PERCUSSIVE ===
            # Percussiveness: Drum-heavy vs melodic - useful for transition selection
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.mean(librosa.feature.rms(y=y_harmonic))
            percussive_energy = np.mean(librosa.feature.rms(y=y_percussive))
            percussiveness = percussive_energy / (harmonic_energy + percussive_energy + 1e-10)

            # === VIBE & CHARACTER FEATURES ===

            # Danceability: Combines beat strength, tempo, and rhythm regularity
            # Higher = more danceable (strong beats + good tempo + regular rhythm)
            tempo_factor = 1.0 if 90 <= tempo_value <= 150 else 0.5  # Optimal dance tempo range
            danceability = (beat_strength * rhythm_regularity * tempo_factor * percussiveness) / 100

            # Acousticness: How acoustic vs electronic (harmonic energy + low spectral flatness = acoustic)
            # High harmonic, low flatness = acoustic instruments; Low harmonic, high flatness = electronic
            acousticness = (harmonic_energy / (energy + 1e-10)) * (1 - spectral_flatness)

            # Instrumentalness: Inverse of vocal presence
            vocal_mask = (freqs >= 2000) & (freqs < 4000)
            vocal_energy = np.mean(stft[vocal_mask, :])
            vocal_presence = vocal_energy / (np.mean(stft) + 1e-10)
            instrumentalness = 1 / (vocal_presence + 0.1)  # Higher = more instrumental

            # Valence (positivity): Major keys and high brightness = positive vibe
            # Major = 1, Minor = 0; brightness weighted
            valence = (0.5 if is_major else 0.0) + (brightness * 0.5)

            # Energy variance: How much the energy fluctuates (dynamic vs steady)
            rms = librosa.feature.rms(y=y)[0]
            energy_variance = np.var(rms)

            # Speechiness: Rhythm regularity + narrow spectral bandwidth in vocal range
            # High = spoken word/rap-like, Low = sung/instrumental
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            speechiness = rhythm_regularity * vocal_presence * (1 / (spectral_bandwidth / 1000))

            # Liveness: Spectral flatness + high frequencies = live/crowd feel
            # Higher = sounds like live recording
            liveness = spectral_flatness * (high_energy / (mid_energy + 1e-10))

            # Warmth: Bass energy relative to total - warm/full vs thin sound
            total_energy = bass_energy + mid_energy + high_energy
            warmth = bass_energy / (total_energy + 1e-10)

            # Aggression: High percussiveness + high energy + high brightness
            aggression = percussiveness * (energy / 0.1) * brightness

            # Complexity: Spectral contrast + harmonic clarity + rhythm regularity
            # How musically complex/interesting vs simple
            complexity = (spectral_contrast / 10) * harmonic_clarity * (rhythm_regularity / 10)

            # === TRACK INFO ===
            duration = librosa.get_duration(y=y, sr=sr)

            # === Update DataFrame ===
            songs_df.at[idx, 'bpm'] = tempo_value
            songs_df.at[idx, 'key'] = key_name
            songs_df.at[idx, 'mode'] = mode
            songs_df.at[idx, 'camelot'] = camelot
            songs_df.at[idx, 'beat_strength'] = beat_strength
            songs_df.at[idx, 'rhythm_regularity'] = rhythm_regularity
            songs_df.at[idx, 'harmonic_clarity'] = harmonic_clarity

            songs_df.at[idx, 'energy'] = energy
            songs_df.at[idx, 'loudness'] = loudness

            songs_df.at[idx, 'bass_energy'] = bass_energy
            songs_df.at[idx, 'mid_energy'] = mid_energy
            songs_df.at[idx, 'high_energy'] = high_energy
            songs_df.at[idx, 'brightness'] = brightness
            songs_df.at[idx, 'spectral_flatness'] = spectral_flatness
            songs_df.at[idx, 'spectral_contrast'] = spectral_contrast

            songs_df.at[idx, 'harmonic_energy'] = harmonic_energy
            songs_df.at[idx, 'percussive_energy'] = percussive_energy
            songs_df.at[idx, 'percussiveness'] = percussiveness

            for i, mfcc_val in enumerate(mfcc_mean):
                songs_df.at[idx, f'mfcc_{i+1}'] = mfcc_val

            # Vibe features
            songs_df.at[idx, 'danceability'] = danceability
            songs_df.at[idx, 'acousticness'] = acousticness
            songs_df.at[idx, 'instrumentalness'] = instrumentalness
            songs_df.at[idx, 'vocal_presence'] = vocal_presence
            songs_df.at[idx, 'valence'] = valence
            songs_df.at[idx, 'energy_variance'] = energy_variance
            songs_df.at[idx, 'speechiness'] = speechiness
            songs_df.at[idx, 'liveness'] = liveness
            songs_df.at[idx, 'warmth'] = warmth
            songs_df.at[idx, 'aggression'] = aggression
            songs_df.at[idx, 'complexity'] = complexity

            songs_df.at[idx, 'analyzed'] = True
            print(f"✓ Processed: {filepath}")

        except Exception as e:
            print(f"✗ Error analyzing {filepath}: {e}")
            continue

    print(f"\nTotal analyzed: {songs_df['analyzed'].sum()}")
    return songs_df


load_dotenv()

playlist_url = os.getenv("PLAYLIST_URL")
download_folder_path = os.getenv("DOWNLOAD_PATH")
DB_PATH = os.getenv("DB_PATH")

# --- INIT DB ---
init_db()

# --- GET PLAYLIST ---
soundcloud_track_info_df = get_playlist_track_info(playlist_url)

# --- CREATE IDS ---
soundcloud_track_info_df["song_id"] = soundcloud_track_info_df.apply(make_song_id, axis=1)

# --- LOAD EXISTING ---
existing_ids = load_existing_ids()

# --- FILTER NEW ---
df_new = soundcloud_track_info_df[
    ~soundcloud_track_info_df["song_id"].isin(existing_ids)
].copy()

print(f"Skipping {len(soundcloud_track_info_df) - len(df_new)} existing songs")
print(f"Processing {len(df_new)} new songs")

if df_new.empty:
    print("Nothing new to process.")
    exit()

# --- DOWNLOAD ---
soundcloud_track_post_dl = download_playlist_files(df_new, download_folder_path)

# --- ANALYZE ---
soundcloud_analyzed_tracks = analyze_downloaded_tracks(soundcloud_track_post_dl, download_folder_path)

# --- SAVE ---
save_to_db(soundcloud_analyzed_tracks)

print("Saved new songs to database.")