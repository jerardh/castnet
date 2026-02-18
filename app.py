from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ---------------------------
# Load model and embeddings
# ---------------------------
model = SentenceTransformer("castnet_finetuned_model_cpu")
embeddings = np.load("castnet_embeddings.npy")

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
# Build age group (mapped to dataset categories)
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
# API ROUTES
# ---------------------------
@app.route("/")
def home():
    return {"status": "CastNet API running"}

@app.route("/predict", methods=["GET", "POST"])
def predict():
    results = []

    if request.method == "POST":
        # ---------------------------
        # Read user input
        # ---------------------------
        user_plot = request.form.get("plot", "").strip()
        user_character_desc = request.form.get("character_description", "").strip()
        user_gender = request.form.get("gender", "").strip().lower()
        raw_age_group = request.form.get("age_group", "").strip().lower()

        # ---------------------------
        # Map UI age group → dataset age group
        # ---------------------------
        AGE_MAP = {
            "0-20": "teen",
            "teen": "teen",

            "20-30": "young",
            "20s": "young",
            "young": "young",

            "30-40": "adult",
            "30s": "adult",
            "adult": "adult",

            "40-50": "middle",
            "40s": "middle",
            "50-60": "middle",
            "50s": "middle",
            "middle": "middle",

            "60+": "senior",
            "senior": "senior"
        }

        user_age_group = AGE_MAP.get(raw_age_group, "")

        print("USER AGE GROUP:", user_age_group)

        # ---------------------------
        # Build query text (same style as training)
        # ---------------------------
        query_text = (
            user_plot +
            ". Character: " + user_character_desc +
            f". Gender: {user_gender}. Age group: {user_age_group}"
        )

        # ---------------------------
        # Encode query
        # ---------------------------
        query_emb = model.encode([query_text])
        scores = cosine_similarity(query_emb, embeddings)[0]

        # ---------------------------
        # Get top candidates
        # ---------------------------
        top_idx = np.argsort(scores)[::-1][:100]
        candidates = df.iloc[top_idx].copy()
        candidates["similarity"] = scores[top_idx]

        # ---------------------------
        # Normalize columns
        # ---------------------------
        candidates["gender"] = candidates["gender"].astype(str).str.strip().str.lower()
        candidates["age_group"] = candidates["age_group"].astype(str).str.strip().str.lower()

        print("DATASET AGE GROUPS:", candidates["age_group"].unique())

        # ---------------------------
        # Apply HARD filters
        # ---------------------------
        if user_gender:
            candidates = candidates[candidates["gender"] == user_gender]

        if user_age_group:
            candidates = candidates[candidates["age_group"] == user_age_group]

        # ---------------------------
        # Final output
        # ---------------------------
        results = candidates.head(5)[
            ["actor_name", "movie_name", "character_name", "gender", "age_group"]
        ].to_dict(orient="records")
    print("AGE GROUP COUNTS=",df["age_group"].value_counts())
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
