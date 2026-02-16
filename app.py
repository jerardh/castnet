from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


embeddings = np.load("final_embeddings.npy")
model = SentenceTransformer("./all-MiniLM-L6-v2")
# Load source files
movies_df = pd.read_csv("movies_with_plot.csv")
roles_df = pd.read_csv("malayalam_movie_cast_dataset.csv")

# Normalize movie names
movies_df["movie_name"] = movies_df["movie_name"].str.strip().str.lower()
roles_df["movie_name"] = roles_df["movie_name"].str.strip().str.lower()

# Merge dynamically
df = pd.merge(
    roles_df,
    movies_df,
    on=["movie_name", "year"],
    how="inner"
)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    user_plot = data.get("plot", "")
    user_character_desc = data.get("character_description", "")

    # Combine same way as training
    query_text = user_plot + " Character: " + user_character_desc

    # Encode
    query_emb = model.encode([query_text])

    # Similarity
    scores = cosine_similarity(query_emb, embeddings)[0]

    top_idx = scores.argsort()[-5:][::-1]
    results = df.iloc[top_idx][["actor_name", "movie_name", "character_name"]]

    return jsonify(results.to_dict(orient="records"))

@app.route("/")
def home():
    return {"status": "API running"}

if __name__ == "__main__":
    app.run(debug=True)
