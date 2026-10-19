"""
Step 3: The core of the project.
- Pulls labelled + unlabelled songs from MySQL
- Runs EDA (liked vs skipped comparison)
- Runs K-means clustering on ALL songs
- Trains Logistic Regression + Random Forest on labelled songs, picks the best
- Uses SHAP to explain predictions
- Predicts on remaining unlabelled songs, writes predictions back to MySQL
- Builds a "Personal Taste Fingerprint" summary + charts

Usage: python 3_taste_analysis.py
"""
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap

DB_PASSWORD = 'root'
FEATURE_COLS = ['tempo', 'energy', 'valence', 'danceability', 'acousticness']
# instrumentalness/loudness/speechiness included if present and not all-null

def get_conn():
    return pymysql.connect(host='localhost', user='root', password=DB_PASSWORD, database='taste_fingerprint')

def load_all_songs():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM songs", conn)
    conn.close()
    return df

def pick_feature_cols(df):
    cols = list(FEATURE_COLS)
    for extra in ['instrumentalness', 'loudness', 'speechiness']:
        if extra in df.columns and df[extra].notna().mean() > 0.8:
            cols.append(extra)
    return cols

def run_clustering(df, feature_cols):
    print("\n--- Step: Clustering (K-Means) ---")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols].fillna(df[feature_cols].mean()))

    inertias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(range(2, 8), inertias, marker='o')
    plt.xlabel('k'); plt.ylabel('Inertia'); plt.title('Elbow Method')
    plt.savefig('elbow.png'); plt.close()

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled)

    summary = df.groupby('cluster')[feature_cols].mean().round(3)
    print(summary)
    return df, scaler, summary

def eda_liked_vs_skipped(df):
    labelled = df[df['label'].notna()]
    if labelled.empty:
        print("No labelled songs yet - skipping EDA comparison.")
        return
    print("\n--- Step: Liked vs Skipped comparison ---")
    comparison = labelled.groupby('label')[FEATURE_COLS].mean().round(3)
    print(comparison)

    fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(4*len(FEATURE_COLS), 4))
    for i, col in enumerate(FEATURE_COLS):
        liked = labelled[labelled['label'] == 1][col]
        skipped = labelled[labelled['label'] == 0][col]
        axes[i].hist([liked, skipped], label=['Liked', 'Skipped'], bins=10, alpha=0.7)
        axes[i].set_title(col)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('liked_vs_skipped.png')
    plt.close()

def train_models(df, feature_cols):
    print("\n--- Step: Training models ---")
    labelled = df[df['label'].notna()].copy()
    if len(labelled) < 20:
        print(f"Only {len(labelled)} labelled songs - need at least 20 to train. "
              "Run the labelling app (2_labeling_app.py) first.")
        return None, None, None

    X = labelled[feature_cols].fillna(labelled[feature_cols].mean())
    y = labelled['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else None
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs) if probs is not None and len(np.unique(y_test)) > 1 else None
        results[name] = {'model': model, 'accuracy': acc, 'precision': prec,
                          'recall': rec, 'f1': f1, 'auc': auc}
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  "
              f"F1: {f1:.3f}  AUC: {auc if auc is None else round(auc,3)}")
        print(f"  Confusion matrix:\n{confusion_matrix(y_test, preds)}")

    best_name = max(results, key=lambda n: results[n]['f1'])
    print(f"\nBest model: {best_name}")
    return results[best_name]['model'], best_name, X.columns.tolist()

def explain_and_predict(df, model, model_name, feature_cols):
    print("\n--- Step: SHAP explanation + predicting remaining songs ---")
    unlabelled = df[df['label'].isna()].copy()
    if unlabelled.empty or model is None:
        print("Nothing to predict, or no model trained yet.")
        return df

    X_unlabelled = unlabelled[feature_cols].fillna(unlabelled[feature_cols].mean())
    probs = model.predict_proba(X_unlabelled)[:, 1]
    unlabelled['prediction_probability'] = probs
    unlabelled['prediction'] = (probs >= 0.5).astype(int)

    # SHAP explanation
    if model_name == 'Random Forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_unlabelled)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:
                # newer SHAP versions return (n_samples, n_features, n_classes)
                sv = sv[:, :, 1]
    else:
        explainer = shap.Explainer(model, X_unlabelled)
        sv = explainer(X_unlabelled).values
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    reasons = []
    for i in range(len(unlabelled)):
        row_shap = sv[i]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:2]
        parts = []
        for idx in top_idx:
            feat = feature_cols[idx]
            direction = "high" if row_shap[idx] > 0 else "low"
            parts.append(f"{direction} {feat}")
        reasons.append(", ".join(parts) + (" -> matches taste" if probs[i] >= 0.5 else " -> differs from taste"))
    unlabelled['reason'] = reasons

    # Write predictions back to MySQL
    conn = get_conn()
    cursor = conn.cursor()
    for _, row in unlabelled.iterrows():
        cursor.execute(
            "UPDATE songs SET prediction=%s, prediction_probability=%s WHERE song_id=%s",
            (int(row['prediction']), float(row['prediction_probability']), row['song_id'])
        )
    conn.commit()
    cursor.close()
    conn.close()

    # Also write cluster back for ALL songs
    conn = get_conn()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("UPDATE songs SET cluster=%s WHERE song_id=%s", (int(row['cluster']), row['song_id']))
    conn.commit()
    cursor.close()
    conn.close()

    print("\nTop predictions (Likely to Like):")
    top = unlabelled.sort_values('prediction_probability', ascending=False).head(10)
    print(top[['song_name', 'artist', 'prediction_probability', 'reason']].to_string(index=False))

    return unlabelled

def build_fingerprint(df):
    print("\n--- Step: Personal Taste Fingerprint ---")
    liked = df[df['label'] == 1]
    if liked.empty:
        print("No liked songs yet.")
        return
    fp = liked[FEATURE_COLS].mean()
    print("\nPERSONAL TASTE FINGERPRINT")
    for feat, val in fp.items():
        bar_len = int(val * 10) if val <= 1 else int(min(val/200,1)*10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        print(f"{feat:15s} {bar}  {val:.2f}")

    liked_clusters = liked['cluster'].value_counts(normalize=True) * 100
    print("\nLiked-song cluster distribution:")
    print(liked_clusters.round(1))

    # Radar chart - normalize each feature to 0-1 using the full dataset's range
    # so tempo (e.g. 50-200) doesn't dwarf features already on a 0-1 scale.
    categories = FEATURE_COLS
    mins = df[categories].min()
    maxs = df[categories].max()
    norm_values = ((fp[categories] - mins) / (maxs - mins)).clip(0, 1).tolist()
    display_values = norm_values + norm_values[:1]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, display_values, 'o-')
    ax.fill(angles, display_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    # Show the real (un-normalized) value in the tick label so it's still readable
    labels = [f"{cat}\n({fp[cat]:.2f})" for cat in categories]
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Personal Taste Fingerprint\n(normalized 0-1 per feature; real values in parentheses)")
    plt.tight_layout()
    plt.savefig('taste_fingerprint.png')
    plt.close()
    print("Saved taste_fingerprint.png")

if __name__ == "__main__":
    df = load_all_songs()
    feature_cols = pick_feature_cols(df)
    print(f"Using features: {feature_cols}")

    df, scaler, cluster_summary = run_clustering(df, feature_cols)
    eda_liked_vs_skipped(df)
    model, model_name, used_cols = train_models(df, feature_cols)
    if model is not None:
        explain_and_predict(df, model, model_name, used_cols)
    build_fingerprint(df)
    print("\nAll done. Charts saved: elbow.png, liked_vs_skipped.png, taste_fingerprint.png")
