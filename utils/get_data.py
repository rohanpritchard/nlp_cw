import string
import nltk
import nltk.tokenize as tokenizer
import numpy as np
from torch import Tensor


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