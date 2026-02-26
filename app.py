from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ---------------------------
# Load CASTING model + embeddings
# ---------------------------
cast_model = SentenceTransformer("castnet_finetuned_model_cpu")
cast_embeddings = np.load("castnet_embeddings.npy")

# ---------------------------
# Load PLOT model + embeddings
# ---------------------------
plot_model = SentenceTransformer("short_plot_similarity_model")
plot_embeddings = np.load("short_plot_embeddings.npy")

# ---------------------------
# Load datasets
# ---------------------------
movies_df = pd.read_csv("movies_with_plot.csv")
roles_df = pd.read_csv("malayalam_movie_cast_dataset.csv")
meta_df = pd.read_csv("actor_metadata.csv")
short_plot_df = pd.read_csv("short_plot_dataset_clean.csv")

# ---------------------------
# Normalize movie names
# ---------------------------
movies_df["movie_name"] = movies_df["movie_name"].str.strip().str.lower()
roles_df["movie_name"] = roles_df["movie_name"].str.strip().str.lower()
short_plot_df["movie_name"] = short_plot_df["movie_name"].str.strip().str.lower()

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

def map_age_to_group(age):
    if age < 20:
        return "teen"
    elif age < 30:
        return "young"
    elif age < 40:
        return "adult"
    elif age < 55:
        return "middle"
    else:
        return "senior"

df["age_group"] = df["age"].apply(map_age_to_group)

# ---------------------------
# Build input text (same as training)
# ---------------------------
df["input_text"] = df["plot"] + " Character: " + df["character_name"].fillna("")

# ---------------------------
# HOME
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
# ---------------------------
# CASTING ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    results = {}

    user_plot = request.form.get("plot", "").strip()

    AGE_MAP = {
        "0-20": "teen", "teen": "teen",
        "20-30": "young", "20s": "young",
        "30-40": "adult", "30s": "adult",
        "40-50": "middle", "40s": "middle",
        "50-60": "middle", "50s": "middle",
        "60+": "senior", "senior": "senior"
    }

    char_indexes = set()
    for key in request.form.keys():
        if key.startswith("char_desc_"):
            char_indexes.add(key.split("_")[-1])

    for idx in sorted(char_indexes, key=int):
        desc = request.form.get(f"char_desc_{idx}", "").strip()
        gender = request.form.get(f"gender_{idx}", "").strip().lower()
        raw_age = request.form.get(f"age_{idx}", "").strip().lower()

        if not desc:
            continue

        user_age_group = AGE_MAP.get(raw_age, "")

        query_text = (
            user_plot +
            ". Character: " + desc +
            f". Gender: {gender}. Age group: {user_age_group}"
        )

        query_emb = cast_model.encode([query_text])
        scores = cosine_similarity(query_emb, cast_embeddings)[0]

        top_idx = np.argsort(scores)[::-1][:100]
        candidates = df.iloc[top_idx].copy()
        candidates["similarity"] = scores[top_idx]

        candidates["gender"] = candidates["gender"].astype(str).str.lower()
        candidates["age_group"] = candidates["age_group"].astype(str).str.lower()

        if gender:
            candidates = candidates[candidates["gender"] == gender]

        if user_age_group:
            candidates = candidates[candidates["age_group"] == user_age_group]

        results[f"Character {idx}"] = candidates.head(5)[
            ["actor_name", "movie_name", "character_name", "gender", "age_group", "similarity"]
        ].to_dict(orient="records")

    return render_template("index.html", results=results)

# ---------------------------
# PLOT SIMILARITY ROUTE
# ---------------------------
@app.route("/plot_similarity", methods=["POST"])
def plot_similarity():
    user_plot = request.form.get("plot", "").strip()

    if not user_plot:
        return render_template(
            "plot_results.html",
            results=[],
            user_plot="",
            message="No plot provided"
        )

    # Encode user plot
    user_emb = plot_model.encode([user_plot])

    # Cosine similarity with dataset
    similarities = cosine_similarity(user_emb, plot_embeddings)[0]

    # Attach similarity to dataframe
    temp_df = short_plot_df.copy()
    temp_df["similarity"] = similarities

    # Sort and take top 5
    top_matches = temp_df.sort_values(
        by="similarity", ascending=False
    ).head(5)

    print("MAX SIM:", top_matches.iloc[0]["similarity"])
    print("TOP 5:", top_matches["similarity"].tolist())

    results = top_matches[
        ["movie_name", "year", "short_plot", "similarity"]
    ].to_dict(orient="records")

    return render_template(
        "plot_results.html",
        results=results,
        user_plot=user_plot
    )
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)