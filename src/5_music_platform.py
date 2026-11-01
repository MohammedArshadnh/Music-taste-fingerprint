"""
A small music platform: search any song in your database, listen to it via
an embedded YouTube player, and click LIKE / SKIP to save your choice to MySQL.

Usage: streamlit run 5_music_platform.py
"""
import streamlit as st
import pymysql
import pandas as pd
import urllib.parse

DB_PASSWORD = 'root'

def get_conn():
    return pymysql.connect(host='localhost', user='root', password=DB_PASSWORD, database='taste_fingerprint')

def search_songs(query):
    conn = get_conn()
    like_query = f"%{query}%"
    df = pd.read_sql(
        "SELECT * FROM songs WHERE song_name LIKE %s OR artist LIKE %s LIMIT 20",
        conn, params=(like_query, like_query)
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

def youtube_embed(song_name, artist):
    """Embeds a YouTube search results player so the song plays inside the app.
    Note: this uses YouTube's unofficial search-embed trick (no API key needed).
    It generally works, but YouTube can change this behavior without notice —
    if the player shows an error, use the fallback link button below it instead."""
    query = urllib.parse.quote(f"{song_name} {artist} audio")
    embed_url = f"https://www.youtube.com/embed?listType=search&list={query}"
    st.components.v1.iframe(embed_url, height=340)

    # Fallback link in case the embed above doesn't load
    search_url = f"https://www.youtube.com/results?search_query={query}"
    st.markdown(f"[If the player above doesn't work, open on YouTube instead]({search_url})")

st.set_page_config(page_title="Music Platform", layout="centered")
st.title("🎵 My Music Platform")

# ---- SEARCH BAR ----
search_query = st.text_input("🔍 Search for a song or artist", "")

if search_query:
    results = search_songs(search_query)
    if results.empty:
        st.warning("No songs found matching that search.")
    else:
        st.caption(f"{len(results)} result(s) found")
        for _, song in results.iterrows():
            with st.container():
                st.subheader(song['song_name'])
                st.write(f"**Artist:** {song['artist']}")
                st.write(f"Tempo: {song['tempo']:.0f} | Energy: {song['energy']:.2f} | "
                         f"Valence: {song['valence']:.2f} | Danceability: {song['danceability']:.2f} | "
                         f"Acousticness: {song['acousticness']:.2f}")

                current_label = song['label']
                if pd.notna(current_label):
                    st.info(f"Currently marked: {'❤️ LIKED' if current_label == 1 else '⏭️ SKIPPED'}")

                youtube_embed(song['song_name'], song['artist'])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("❤️ LIKE", key=f"like_{song['song_id']}", use_container_width=True):
                        save_label(song['song_id'], 1)
                        st.success("Saved as LIKED")
                        st.rerun()
                with col2:
                    if st.button("⏭️ SKIP", key=f"skip_{song['song_id']}", use_container_width=True):
                        save_label(song['song_id'], 0)
                        st.success("Saved as SKIPPED")
                        st.rerun()
                st.divider()
else:
    st.write("Type a song name or artist above to search your database.")

    conn = get_conn()
    total = pd.read_sql("SELECT COUNT(*) as c FROM songs", conn)['c'][0]
    labelled = pd.read_sql("SELECT COUNT(*) as c FROM songs WHERE label IS NOT NULL", conn)['c'][0]
    conn.close()
    st.caption(f"Database: {total} songs total, {labelled} labelled so far.")
