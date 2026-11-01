"""
Step 2: Streamlit app for Like/Skip labelling.

Usage: streamlit run 2_labeling_app.py
(Do NOT run this with plain 'python' - it must be run with the streamlit command.)
"""
import streamlit as st
import pymysql
import pandas as pd

DB_PASSWORD = 'root'  # <-- your MySQL root password

def get_conn():
    return pymysql.connect(host='localhost', user='root', password=DB_PASSWORD, database='taste_fingerprint')

def load_unlabelled_sample():
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM songs WHERE selected_for_labelling = 1 AND label IS NULL", conn
    )
    conn.close()
    return df

def save_label(song_id, label):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE songs SET label = %s WHERE song_id = %s", (label, song_id))
    conn.commit()
    cursor.close()
    conn.close()

st.set_page_config(page_title="Personal Taste Labelling", layout="centered")
st.title("🎵 Like or Skip?")

if 'queue' not in st.session_state:
    st.session_state.queue = load_unlabelled_sample()

# progress
conn = get_conn()
total = pd.read_sql("SELECT COUNT(*) as c FROM songs WHERE selected_for_labelling = 1", conn)['c'][0]
done = pd.read_sql("SELECT COUNT(*) as c FROM songs WHERE selected_for_labelling = 1 AND label IS NOT NULL", conn)['c'][0]
conn.close()
st.progress(done / total if total else 0)
st.caption(f"{done} / {total} labelled")

queue = st.session_state.queue
if len(queue) == 0:
    st.success("All songs in the sample have been labelled! You can now run 3_taste_analysis.py")
else:
    song = queue.iloc[0]
    st.subheader(f"{song['song_name']}")
    st.write(f"**Artist:** {song['artist']}")
    st.write(f"Tempo: {song['tempo']:.0f} | Energy: {song['energy']:.2f} | "
             f"Valence: {song['valence']:.2f} | Danceability: {song['danceability']:.2f} | "
             f"Acousticness: {song['acousticness']:.2f}")

    import urllib.parse
    search_query = urllib.parse.quote(f"{song['song_name']} {song['artist']}")
    youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
    st.link_button("🎧 Listen on YouTube", youtube_url)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("❤️ LIKE", use_container_width=True):
            save_label(song['song_id'], 1)
            st.session_state.queue = queue.iloc[1:].reset_index(drop=True)
            st.rerun()
    with col2:
        if st.button("⏭️ SKIP", use_container_width=True):
            save_label(song['song_id'], 0)
            st.session_state.queue = queue.iloc[1:].reset_index(drop=True)
            st.rerun()
