from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

import os 
import pickle
import faiss
import numpy as np
from tqdm import tqdm


def build_embeddings(model_path: str, 
                train_set, 
                test_set, 
                n_clusters: int=256, 
                embedding_size: int=768, 
                nprobe: int=4):
    
    # load model
    model = SentenceTransformer(model_path)

    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    corpus_sentences = []
    corpus_label = []
    idx2label = {}
    for idx, label in enumerate(train_set.target_names):
        idx2label[idx] = label
    
    for sent, target in zip(train_set.data, train_set.target):
        corpus_sentences.append(sent)
        corpus_label.append(target)
    
    corpus_embedding = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    embeddings_path = '/content/drive/MyDrive/vocads_challenge/embeddings/embeddings.pkl'
    with open(embeddings_path, 'wb') as fOut:
        pickle.dump({'embeddings': corpus_embedding, 'targets': corpus_label, 'label': idx2label}, fOut)

if __name__ == "__main__":

    model_path = "/content/drive/MyDrive/vocads_challenge/model"

    train_set = fetch_20newsgroups(subset="train", remove=('headers', 'footers','quotes'))
    test_set = fetch_20newsgroups(subset='test', remove=('headers', 'footers','quotes')) 

    build_embeddings(model_path, 
                    train_set, 
                    test_set)












