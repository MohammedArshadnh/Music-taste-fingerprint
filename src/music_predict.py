"""
Terminal-based search: type a song name, see its 5 audio features,
and get a Like/Skip prediction with percentage.

Usage: python search_predict.py
"""
import os
import pandas as pd
import numpy as np
import pymysql
from sklearn.ensemble import RandomForestClassifier

DB_PASSWORD = os.getenv("DB_PASSWORD")
FEATURE_COLS = ['tempo', 'energy', 'valence', 'danceability', 'acousticness']

def get_conn():
    if not DB_PASSWORD:
        raise ValueError("DB_PASSWORD is not set.")
    return pymysql.connect(host='localhost', user='root', password=DB_PASSWORD, database='taste_fingerprint')

def train_model():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM songs WHERE label IS NOT NULL", conn)
    conn.close()
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())
    y = df['label'].astype(int)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, len(df)

def search_songs(query):
    conn = get_conn()
    like_query = f"%{query}%"
    df = pd.read_sql(
        "SELECT * FROM songs WHERE song_name LIKE %s OR artist LIKE %s LIMIT 15",
        conn, params=(like_query, like_query)
    )
    conn.close()
    return df

def show_song_result(song, model):
    print("\n" + "=" * 60)
    print(f"Song:   {song['song_name']}")
    print(f"Artist: {song['artist']}")
    print("-" * 60)
    print("FEATURES:")
    for col in FEATURE_COLS:
        print(f"  {col:15s}: {song[col]:.2f}" if col != 'tempo' else f"  {col:15s}: {song[col]:.0f} BPM")

    print("-" * 60)
    if pd.notna(song['label']):
        actual = "YOU LIKED THIS" if song['label'] == 1 else "YOU SKIPPED THIS"
        print(f"RESULT: {actual} (you already labelled this song)")
    else:
        features = pd.DataFrame([song[FEATURE_COLS]])
        prob = model.predict_proba(features)[0][1]
        like_pct = prob * 100
        skip_pct = 100 - like_pct
        verdict = "LIKELY TO LIKE" if prob >= 0.5 else "LIKELY TO SKIP"
        print(f"PREDICTION: {verdict}")
        print(f"  Like %: {like_pct:.1f}%")
        print(f"  Skip %: {skip_pct:.1f}%")
    print("=" * 60)

def save_new_song(name, artist, tempo, energy, valence, danceability, acousticness, label=None):
    import uuid
    song_id = "manual_" + uuid.uuid4().hex[:10]
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO songs (song_id, song_name, artist, tempo, energy, valence,
                            danceability, acousticness, label, selected_for_labelling)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
    """, (song_id, name, artist, tempo, energy, valence, danceability, acousticness, label))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved '{name}' to your database (song_id: {song_id}).")

def manual_entry_prediction(model):
    print("\nThis song isn't in your database. Look it up on tunebat.com and enter its values below.")
    name = input("Song name: ").strip()
    artist = input("Artist: ").strip()
    try:
        tempo = float(input("Tempo (BPM, e.g. 120): "))
        energy = float(input("Energy (0-1, e.g. 0.75): "))
        valence = float(input("Valence/happiness (0-1, e.g. 0.5): "))
        danceability = float(input("Danceability (0-1, e.g. 0.6): "))
        acousticness = float(input("Acousticness (0-1, e.g. 0.2): "))
    except ValueError:
        print("That wasn't a valid number - cancelling.")
        return

    song = pd.Series({
        'song_name': name, 'artist': artist, 'tempo': tempo, 'energy': energy,
        'valence': valence, 'danceability': danceability, 'acousticness': acousticness,
        'label': np.nan
    })
    show_song_result(song, model)

    save = input("\nSave this song to your database? (y/n): ").strip().lower()
    if save == 'y':
        label_input = input("Do you want to label it now too? (like/skip/skip this step): ").strip().lower()
        label = None
        if label_input == 'like':
            label = 1
        elif label_input == 'skip':
            label = 0
        save_new_song(name, artist, tempo, energy, valence, danceability, acousticness, label)

def main():
    print("Training model on your current labelled songs...")
    model, n_labelled = train_model()
    print(f"Model ready (trained on {n_labelled} labelled songs).\n")

    while True:
        query = input("\nSearch for a song or artist (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            continue

        results = search_songs(query)
        if results.empty:
            print("No songs found matching that search in your database.")
            add_new = input("Is this a NEW song you want to check? (y/n): ").strip().lower()
            if add_new == 'y':
                manual_entry_prediction(model)
            continue

        print(f"\nFound {len(results)} result(s):")
        for i, row in results.iterrows():
            print(f"  [{i}] {row['song_name']} - {row['artist']}")

        choice = input("\nEnter the number to view details (or press Enter to search again): ").strip()
        if not choice:
            continue
        if choice.isdigit() and int(choice) in results.index:
            show_song_result(results.loc[int(choice)], model)
        else:
            valid = ", ".join(str(i) for i in results.index)
            print(f"'{choice}' isn't a valid number. Please enter one of: {valid}")

if __name__ == "__main__":
    main()
