# Music Taste Fingerprint 

## About the Project
Music Taste Fingerprint is a Machine Learning project that analyzes a user's music preferences and predicts whether a song matches their taste.

## Technologies Used
- Python
- MySQL
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit

## Features
- Load and manage song datasets
- Label songs as liked or skipped
- Analyze music features
- Predict music preferences using Machine Learning
- Search songs by name or artist
- Export results

## Machine Learning Algorithm
Random Forest Classifier

The model uses features such as tempo, energy, valence, danceability, and acousticness to predict music preferences.

## Project Files
- `1_load_data.py` – Loads music data
- `2_labeling_app.py` – Labels songs
- `3_taste_analysis.py` – Analyzes music preferences
- `5_music_platform.py` – Music platform interface
- `music_predict.py` – Machine Learning prediction
- `export_results.py` – Exports results
- `my_songs.csv` – Song dataset
- `liked_vs_skipped.png` – Analysis visualization
- `taste_fingerprint.png` – Project visualization

## Installation

Install the required libraries:

```bash
pip install pandas numpy pymysql scikit-learn streamlit
```

## Configuration
Configure your local MySQL database and set the `DB_PASSWORD` environment variable before running the application.

**Important:** Never upload your database password or other secret credentials to GitHub.

## Author
Mohammed Arshad NH
