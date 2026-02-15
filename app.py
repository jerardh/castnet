from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

df = pd.read_csv("final_dataset_with_text.csv")
embeddings = np.load("plot_embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_plot = data["plot"]

    query_emb = model.encode([user_plot])
    scores = cosine_similarity(query_emb, embeddings)[0]

    top_idx = scores.argsort()[-5:][::-1]
    results = df.iloc[top_idx][["actor_name", "movie_name", "character_name"]]

    return jsonify(results.to_dict(orient="records"))

@app.route("/")
def home():
    return {"status": "API running"}


if __name__ == "__main__":
    app.run(debug=True)
