from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
# from flask_ngrok import run_with_ngrok

import os 
import pickle
import faiss
import numpy as np

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
# run_with_ngrok(app)  # Start ngrok when app is run

model_path = "/content/drive/MyDrive/vocads_challenge/model"
embeddings_path = '/content/drive/MyDrive/vocads_challenge/embeddings/embeddings.pkl'

index_nprobe = 4
embedding_size = 768
top_k = 10
n_cluster = 256

model = SentenceTransformer(model_path)

quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_cluster, faiss.METRIC_INNER_PRODUCT)
index.n_probe = index_nprobe

with open(embeddings_path, 'rb') as fIn:
    cache_data = pickle.load(fIn)
    corpus_embedding = cache_data['embeddings']
    corpus_label = cache_data['targets']
    idx2label = cache_data['label']

corpus_embedding = corpus_embedding / np.linalg.norm(corpus_embedding, axis=1)[:, None]
index.train(corpus_embedding)
index.add(corpus_embedding)

@app.route("/predict", methods=["GET"])
def predict():
    
    query = request.args.get("query")
    embedding = model.encode(query)
    embedding = embedding / np.linalg.norm(embedding)
    distances, corpus_ids = index.search(embedding, top_k)

    hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    label_idx = corpus_label[hits[0]['corpus_id']]
    
    return(jsonify(query=query,label=str(idx2label[label_idx]))) 


if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0')
