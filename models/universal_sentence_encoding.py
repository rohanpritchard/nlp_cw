import tensorflow_hub as hub
import numpy as np
import tensorflow_text

def get_similarity(eng_sent, ger_sent):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    eng_embeddings = embed(eng_sent) 
    ger_embeddings = embed(ger_sent) 
    similarity_matrix = np.inner(eng_embeddings, ger_embeddings)
    return np.diag(similarity_matrix)


