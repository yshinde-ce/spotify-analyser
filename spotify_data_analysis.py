import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set_theme(style="whitegrid")
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: Download the Spotify dataset if it is not already present
CSV_PATH = "dataset.csv"
DATA_URL = (
    "https://huggingface.co/datasets/"
    "maharshipandya/spotify-tracks-dataset/resolve/main/"
    "dataset.csv"
)

if not os.path.exists(CSV_PATH):
    print("Downloading Spotify dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, CSV_PATH)
        print("Dataset downloaded successfully!")
    except Exception as e:
        raise RuntimeError(
            "Could not download the dataset. Download dataset.csv from "
            "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset "
            "and place it in the same folder as this Python file. "
            f"Original error: {e}"
        )

# Step 2: Load and inspect the data
df = pd.read_csv(CSV_PATH)
print("\nOriginal dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum()[lambda x: x > 0])

# Check required columns
required = [
    "track_id", "track_name", "artists", "album_name", "track_genre",
    "popularity", "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]
missing_columns = [column for column in required if column not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Step 3: Clean the data
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

text_cols = ["track_id", "track_name", "artists", "album_name", "track_genre"]
df = df.dropna(subset=text_cols).copy()

numeric_cols = [
    "popularity", "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]
for column in numeric_cols:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=numeric_cols).copy()
df = df[(df["duration_ms"] > 0) & df["popularity"].between(0, 100)].copy()
df["duration_min"] = df["duration_ms"] / 60000

# Use one row per track/genre for genre analysis and one row per track overall
df_genre = df.drop_duplicates(subset=["track_id", "track_genre"]).copy()
df_unique = df.drop_duplicates(subset=["track_id"]).copy()

if df_unique.empty or df_genre.empty:
    raise ValueError("No valid records remain after cleaning the dataset.")

print("\nCleaned rows:", len(df))
print("Unique tracks:", len(df_unique))
print("Number of genres:", df_genre["track_genre"].nunique())

# Step 4: Descriptive statistics
print("\n========== DESCRIPTIVE STATISTICS ==========")
cols = [
    "popularity", "duration_min", "danceability", "energy",
    "valence", "tempo", "loudness"
]
print(df_unique[cols].describe().round(2))

pop = df_unique["popularity"].to_numpy()
dur = df_unique["duration_min"].to_numpy()

print("\nMean popularity:", round(float(np.mean(pop)), 2))
print("Median popularity:", round(float(np.median(pop)), 2))
print("Mean duration (minutes):", round(float(np.mean(dur)), 2))
print("Median duration (minutes):", round(float(np.median(dur)), 2))

summary_table = pd.DataFrame({
    "Popularity": [
        pop.mean(), np.median(pop), pop.std(ddof=1), pop.min(), pop.max()
    ],
    "Duration (minutes)": [
        dur.mean(), np.median(dur), dur.std(ddof=1), dur.min(), dur.max()
    ],
    "Danceability": [
        df_unique["danceability"].mean(), df_unique["danceability"].median(),
        df_unique["danceability"].std(), df_unique["danceability"].min(),
        df_unique["danceability"].max()
    ],
    "Energy": [
        df_unique["energy"].mean(), df_unique["energy"].median(),
        df_unique["energy"].std(), df_unique["energy"].min(),
        df_unique["energy"].max()
    ]
}, index=["Mean", "Median", "Std deviation", "Minimum", "Maximum"]).round(3)

print("\nSummary statistics table:")
print(summary_table)
summary_table.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"))

# Step 5: Genre-wise analysis
print("\n========== GENRE ANALYSIS ==========")
genre_summary = (
    df_genre.groupby("track_genre")
    .agg(
        tracks=("track_id", "count"),
        avg_popularity=("popularity", "mean"),
        avg_danceability=("danceability", "mean"),
        avg_duration_min=("duration_min", "mean")
    )
    .round(2)
    .sort_values("avg_popularity", ascending=False)
)
print("\nTop 10 genres:")
print(genre_summary.head(10))
print("\nBottom 10 genres:")
print(genre_summary.tail(10))
genre_summary.to_csv(os.path.join(OUT_DIR, "genre_summary.csv"))

# Step 6: Correlation analysis
print("\n========== CORRELATION ANALYSIS ==========")
features = [
    "popularity", "duration_min", "danceability", "energy",
    "valence", "tempo", "loudness", "acousticness"
]
corr = df_unique[features].corr()
print("\nCorrelation with popularity:")
print(corr["popularity"].sort_values(ascending=False).round(3))

x = df_unique["danceability"].to_numpy()
y = pop
denominator = np.sqrt(
    np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2)
)
r_manual = (
    np.sum((x - x.mean()) * (y - y.mean())) / denominator
    if denominator != 0 else np.nan
)
print("\nManual Pearson correlation (danceability vs popularity):",
      round(float(r_manual), 4) if np.isfinite(r_manual) else "undefined")

# Step 7.1: Bar chart of top 15 genres
top15 = genre_summary.head(15).sort_values("avg_popularity")
plt.figure(figsize=(10, 6))
plt.barh(top15.index, top15["avg_popularity"], color="#1DB954")
plt.title("Top 15 Genres by Average Popularity")
plt.xlabel("Average Popularity (0-100)")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_top_genres.png"), dpi=150)
plt.show()

# Step 7.2: Histogram of song duration
plt.figure(figsize=(9, 5))
plt.hist(dur.clip(max=10), bins=40, color="#1DB954", edgecolor="black")
plt.axvline(np.mean(dur), color="red", linestyle="--", label="Mean")
plt.axvline(np.median(dur), color="blue", linestyle="--", label="Median")
plt.title("Distribution of Song Duration")
plt.xlabel("Duration in Minutes (capped at 10 for display)")
plt.ylabel("Number of Tracks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_duration_histogram.png"), dpi=150)
plt.show()

# Step 7.3: Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_correlation_heatmap.png"), dpi=150)
plt.show()

# Step 7.4: Scatter plot
sample = df_unique.sample(n=min(5000, len(df_unique)), random_state=42)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=sample, x="danceability", y="popularity",
    alpha=0.3, color="#1DB954"
)
plt.title("Danceability vs Popularity")
plt.xlabel("Danceability (0-1)")
plt.ylabel("Popularity (0-100)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_scatter.png"), dpi=150)
plt.show()

# Step 7.5: Box plot of popularity in top 10 genres
top_genres = genre_summary.head(10).index
box_data = df_genre[df_genre["track_genre"].isin(top_genres)]
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=box_data, x="track_genre", y="popularity",
    hue="track_genre", palette="viridis", legend=False
)
plt.xticks(rotation=45, ha="right")
plt.title("Popularity Distribution Across Top 10 Genres")
plt.xlabel("Genre")
plt.ylabel("Popularity (0-100)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_popularity_boxplot.png"), dpi=150)
plt.show()

# Step 8: Findings
print("\n========== FINDINGS ==========")
print("\nHighest average popularity genres:")
print(genre_summary.head(5)["avg_popularity"])
print("\nLowest average popularity genres:")
print(genre_summary.tail(3)["avg_popularity"])
print("\nPercentage of tracks with popularity zero:",
      round(float((pop == 0).mean() * 100), 2), "%")
print("Danceability-popularity correlation:",
      round(float(corr.loc["danceability", "popularity"]), 3))
print("Energy-loudness correlation:",
      round(float(corr.loc["energy", "loudness"]), 3))
print("Mean duration:", round(float(np.mean(dur)), 2), "minutes")
print("Median duration:", round(float(np.median(dur)), 2), "minutes")

q1, q3 = np.percentile(dur, [25, 75])
share_2_5_to_5 = ((dur >= 2.5) & (dur <= 5)).mean() * 100
print(f"Tracks lasting 2.5-5 minutes: {share_2_5_to_5:.1f}%")
print(f"Middle 50% of track durations: {q1:.2f}-{q3:.2f} minutes")
print("\nLimitations:")
print("- Popularity is a snapshot at the time the dataset was collected.")
print("- Correlation does not imply causation.")
print("- The dataset may not represent all Spotify listeners or regions.")

print(f"\nCompleted! Charts and CSV summaries are saved in '{OUT_DIR}'.")
