from flask import Flask, request, jsonify
import faiss, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load mô hình và dữ liệu đã index
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
index = faiss.read_index("real_estate_combined_index.faiss")
df = pd.read_csv("real_estate_combined_text.csv")

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    q = data.get("query","").strip()
    if not q:
        return jsonify({"error":"Missing 'query'"}),400
    emb = model.encode([q])
    D, I = index.search(np.array(emb), 3)
    results = df.iloc[I[0]]['text'].tolist()
    return jsonify({"query":q,"results":results})

if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
