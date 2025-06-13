from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import gdown
import os

app = Flask(__name__)

# Các file cần tải
FAISS_FILE_ID = "1wp9-SORgE3ca7IbBTFbs0APz-5Age6ad"
CSV_FILE_ID = "1TAnoR0_Epl_7NN8zMehheH21NI1SEeJr"

# Tải file nếu chưa tồn tại
if not os.path.exists("real_estate_combined_index.faiss"):
    gdown.download(f"https://drive.google.com/uc?id={FAISS_FILE_ID}", "real_estate_combined_index.faiss", quiet=False)

if not os.path.exists("real_estate_combined_text.csv"):
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", "real_estate_combined_text.csv", quiet=False)

# Load mô hình và dữ liệu
print("Đang load mô hình và dữ liệu...")
model = SentenceTransformer("keepitreal/vietnamese-sbert-base")
df = pd.read_csv("real_estate_combined_text.csv")
index = faiss.read_index("real_estate_combined_index.faiss")

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    results = df.iloc[I[0]].to_dict(orient="records")
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)