import string
import fasttext
import fasttext.util
import pickle

import nltk
import nltk.tokenize as tokenizer
import numpy as np
from torch import Tensor
import spacy

def get_data(set):
    source = "./en-de/%s.ende.src" % set
    mt = "./en-de/%s.ende.mt" % set
    scores = "./en-de/%s.ende.scores" % set
    with open(source, "r", encoding='utf-8') as source, open(mt, "r", encoding='utf-8') as mt, open(scores, "r", encoding='utf-8') as scores:
        return list(zip(source.readlines(), mt.readlines(), [float(i) for i in scores.readlines()]))

def get_word_to_index(sentences):
    vocab = set()
    for s in sentences:
        for word in tokenizer.word_tokenize(s):
            vocab.add(word.lower())
    vocab = dict([(y, x+2) for x, y in enumerate(vocab)])
    vocab['<PAD>'] = 0
    vocab['<START>'] = 1
    return vocab

def tokenize(sentence, word_to_index):
    # Append all sentences with <START> token to account for empty sentences.
    return [word_to_index['<START>']] + [word_to_index[w] for w in tokenizer.word_tokenize(sentence) if w in word_to_index]

def normalize(sentences):
    # Takes list of list of tokens
    maximum = max([len(s) for s in sentences])
    return zip(*[(s + ([0]*(maximum-len(s))), len(s)) for s in sentences])

def normalize_embeddings(sentences):
    maximum = max([len(s) for s in sentences])

    array = np.zeros((len(sentences), maximum, 300), dtype=np.float)
    lens = []
    for i, s in enumerate(sentences):
        lens.append(len(s))
        for j, t in enumerate(s):
            array[i,j] = t
    return array, lens
    #return zip(*[(s + ([[0]*300]*(maximum-len(s))), len(s)) for s in sentences])

class Embedder:
    def __init__(self):
        self.en = spacy.load("en_core_web_md")
        self.ge = spacy.load("de_core_news_md")

    def embed_en(self, sentence):
        return [t.vector for t in self.en(sentence)]

    def embed_ge(self, sentence):
        return [t.vector for t in self.ge(sentence)]

'''
print("Loading embedder...")
embedder = Embedder()

print("Getting data...")
data = get_data("train")
data = [(embedder.embed_en(eng), embedder.embed_ge(ger), score) for eng, ger, score in data]
with open("embedded_data.txt", 'wb') as f:
    pickle.dump(data, f)
'''