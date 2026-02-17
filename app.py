from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ---------------------------
# Load embeddings and model
# ---------------------------
embeddings = np.load("final_embeddings.npy")
model = SentenceTransformer("./all-MiniLM-L6-v2")

# ---------------------------
# Load datasets
# ---------------------------
movies_df = pd.read_csv("movies_with_plot.csv")
roles_df = pd.read_csv("malayalam_movie_cast_dataset.csv")
meta_df = pd.read_csv("actor_metadata.csv")

# ---------------------------
# Normalize movie names
# ---------------------------
movies_df["movie_name"] = movies_df["movie_name"].str.strip().str.lower()
roles_df["movie_name"] = roles_df["movie_name"].str.strip().str.lower()

# ---------------------------
# Merge roles + movies
# ---------------------------
df = pd.merge(
    roles_df,
    movies_df,
    on=["movie_name", "year"],
    how="inner"
)

# ---------------------------
# Merge actor metadata
# ---------------------------
df = pd.merge(
    df,
    meta_df,
    on="actor_name",
    how="left"
)

# ---------------------------
# Clean gender
# ---------------------------
df["gender"] = df["gender"].fillna("unknown").str.strip().str.lower()

# ---------------------------
# Build age group
# ---------------------------
CURRENT_YEAR = 2026
df["age"] = CURRENT_YEAR - df["birth_year"]
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 20, 30, 40, 50, 60, 100],
    labels=["teen", "20s", "30s", "40s", "50s", "60+"]
)
df["age_group"] = df["age_group"].astype(str).fillna("unknown")

# ---------------------------
# Build input text (same as training)
# ---------------------------
df["input_text"] = df["plot"] + " Character: " + df["character_name"].fillna("")

# ---------------------------
# API ROUTES
# ---------------------------
@app.route("/")
def home():
    return {"status": "CastNet API running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    user_plot = data.get("plot", "")
    user_character_desc = data.get("character_description", "")
    user_gender = data.get("gender", "").lower().strip()
    user_age_group = data.get("age_group", "").lower().strip()

    query_text = user_plot + " Character: " + user_character_desc
    query_emb = model.encode([query_text])

    scores = cosine_similarity(query_emb, embeddings)[0]

    top_idx = np.argsort(scores)[::-1][:100]
    candidates = df.iloc[top_idx].copy()
    candidates["similarity"] = scores[top_idx]

    # Normalize dataset fields
    candidates["gender"] = candidates["gender"].astype(str).str.lower()
    candidates["age_group"] = candidates["age_group"].astype(str).str.lower()

    # Apply filters only if provided
    if user_gender:
        filtered = candidates[candidates["gender"] == user_gender]
        if not filtered.empty:
            candidates = filtered

    if user_age_group:
        filtered = candidates[candidates["age_group"] == user_age_group]
        if not filtered.empty:
            candidates = filtered

    results = candidates.head(5)[
        ["actor_name", "movie_name", "character_name", "gender", "age_group"]
    ]

    return jsonify(results.to_dict(orient="records"))



if __name__ == "__main__":
    app.run(debug=True)
