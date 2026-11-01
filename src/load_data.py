"""
Automated Step 1: Loads your own liked songs (my_songs.csv) AND the Kaggle
dataset (train.csv) into MySQL in one go, then randomly selects 200 Kaggle
songs for labelling. Run this single script instead of doing things manually.

Usage: python load_data.py
"""
import pandas as pd
import pymysql
import os

DB_PASSWORD = 'root'   # <-- your MySQL root password
KAGGLE_CSV = 'train.csv'
MY_SONGS_CSV = 'data/my_songs.csv'
SAMPLE_SIZE = 200
MAX_KAGGLE_SONGS = 1000

COLUMN_MAP = {
    'track_id': 'song_id', 'track_name': 'song_name', 'artists': 'artist',
    'tempo': 'tempo', 'energy': 'energy', 'valence': 'valence',
    'danceability': 'danceability', 'acousticness': 'acousticness',
    'instrumentalness': 'instrumentalness', 'loudness': 'loudness',
    'speechiness': 'speechiness', 'duration_ms': 'duration_ms', 'track_genre': 'genre',
}

def get_conn():
    return pymysql.connect(host='localhost', user='root', password=DB_PASSWORD, database='taste_fingerprint')

def load_my_songs(conn):
    if not os.path.exists(MY_SONGS_CSV):
        print(f"[1/3] {MY_SONGS_CSV} not found - skipping your personal songs.")
        return 0
    df = pd.read_csv(MY_SONGS_CSV)
    df = df.dropna(subset=['song_id', 'song_name', 'artist', 'tempo', 'energy',
                            'valence', 'danceability', 'acousticness'])
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO songs (song_id, song_name, artist, tempo, energy, valence,
                                danceability, acousticness, genre, label, selected_for_labelling)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,1,1)
            ON DUPLICATE KEY UPDATE label=1, selected_for_labelling=1
        """, (row['song_id'], row['song_name'], row['artist'], row['tempo'], row['energy'],
              row['valence'], row['danceability'], row['acousticness'], row.get('genre')))
    conn.commit()
    cursor.close()
    print(f"[1/3] Loaded {len(df)} of your own songs (auto-marked as label=1, liked).")
    return len(df)

def load_kaggle_songs(conn):
    if not os.path.exists(KAGGLE_CSV):
        print(f"[2/3] {KAGGLE_CSV} not found - skipping Kaggle dataset.")
        return 0
    df = pd.read_csv(KAGGLE_CSV)
    available = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    for col in ['instrumentalness', 'loudness', 'speechiness', 'duration_ms', 'genre']:
        if col not in df.columns:
            df[col] = None
    df = df.dropna(subset=['song_id', 'song_name', 'artist', 'tempo', 'energy',
                            'valence', 'danceability', 'acousticness'])
    df = df.drop_duplicates(subset='song_id').head(MAX_KAGGLE_SONGS)

    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO songs (song_id, song_name, artist, tempo, energy, valence,
                                danceability, acousticness, instrumentalness, loudness,
                                speechiness, duration_ms, genre)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE song_name=song_name
        """, (row['song_id'], row['song_name'], row['artist'], row['tempo'], row['energy'],
              row['valence'], row['danceability'], row['acousticness'],
              row.get('instrumentalness'), row.get('loudness'), row.get('speechiness'),
              row.get('duration_ms'), row.get('genre')))
    conn.commit()
    cursor.close()
    print(f"[2/3] Loaded {len(df)} Kaggle songs.")
    return df['song_id'].tolist()

def select_labelling_sample(conn, kaggle_ids):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM songs WHERE selected_for_labelling = 1 AND label IS NULL")
    already = cursor.fetchone()[0]
    if already > 0:
        print(f"[3/3] {already} Kaggle songs already selected for labelling from a previous run.")
        cursor.close()
        return
    if not kaggle_ids:
        print("[3/3] No Kaggle songs available to select for labelling.")
        cursor.close()
        return
    import random
    random.seed(42)
    sample = random.sample(kaggle_ids, min(SAMPLE_SIZE, len(kaggle_ids)))
    for sid in sample:
        cursor.execute("UPDATE songs SET selected_for_labelling = 1 WHERE song_id = %s", (sid,))
    conn.commit()
    cursor.close()
    print(f"[3/3] Randomly selected {len(sample)} Kaggle songs for you to label.")

def main():
    conn = get_conn()
    load_my_songs(conn)
    kaggle_ids = load_kaggle_songs(conn)
    select_labelling_sample(conn, kaggle_ids)
    conn.close()
    print("\nAll done automatically!")
    print("Next step: streamlit run 2_labeling_app.py")

if __name__ == "__main__":
    main()
