"""
Exports a clean, readable results file showing every song's 5 audio
features plus its Like % and Skip % prediction.

Usage: python export_results.py
Output: prediction_results.csv (open in Excel) + prints a preview in the terminal
"""
import pandas as pd
import pymysql

DB_PASSWORD = 'root'

def main():
    conn = pymysql.connect(host='localhost', user='root', password=DB_PASSWORD, database='taste_fingerprint')
    df = pd.read_sql("""
        SELECT song_name, artist, tempo, energy, valence, danceability, acousticness,
               label, prediction, prediction_probability, cluster
        FROM songs
        ORDER BY prediction_probability DESC
    """, conn)
    conn.close()

    # Build Like % / Skip % columns
    df['like_percent'] = (df['prediction_probability'] * 100).round(1)
    df['skip_percent'] = (100 - df['like_percent']).round(1)

    def status(row):
        if pd.notna(row['label']):
            return "YOU LIKED IT" if row['label'] == 1 else "YOU SKIPPED IT"
        elif pd.notna(row['prediction']):
            return "PREDICTED: LIKE" if row['prediction'] == 1 else "PREDICTED: SKIP"
        else:
            return "NOT YET ANALYSED"
    df['status'] = df.apply(status, axis=1)

    output = df[['song_name', 'artist', 'tempo', 'energy', 'valence', 'danceability',
                 'acousticness', 'like_percent', 'skip_percent', 'status', 'cluster']]
    output.columns = ['Song', 'Artist', 'Tempo', 'Energy', 'Valence', 'Danceability',
                       'Acousticness', 'Like %', 'Skip %', 'Status', 'Cluster']

    output.to_csv('prediction_results.csv', index=False)
    print(f"Saved {len(output)} songs to prediction_results.csv\n")

    print("=== YOUR ACTUAL LABELLED SONGS (sample) ===")
    labelled = output[output['Status'].str.contains("YOU")]
    print(labelled.head(10).to_string(index=False))

    print("\n=== TOP PREDICTED 'LIKELY TO LIKE' SONGS ===")
    predicted = output[output['Status'] == "PREDICTED: LIKE"].sort_values('Like %', ascending=False)
    print(predicted.head(10).to_string(index=False))

    print("\n=== TOP PREDICTED 'LIKELY TO SKIP' SONGS ===")
    skipped = output[output['Status'] == "PREDICTED: SKIP"].sort_values('Skip %', ascending=False)
    print(skipped.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
