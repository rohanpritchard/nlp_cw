import tensorflow_hub as hub
import numpy as np
import tensorflow_text

def get_similarity(eng_sent, ger_sent):
    eng_embeddings = embed(eng_sent) 
    ger_embeddings = embed(ger_sent) 
    similarity_matrix = np.inner(eng_embeddings, ger_embeddings)
    return np.diag(similarity_matrix)

def get_embedding(sentence):
    return embed(sentence)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
