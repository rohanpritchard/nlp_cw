import atexit
import pickle
import subprocess
from os import path
import os

import nltk.tokenize as tokenizer



resources_dir = "./resources"

class UnsupportedEmbeddingError(ValueError):
  pass
class UnsupportedLanguageError(ValueError):
  pass


def get_data(set, language):
  source = path.join(resources_dir, "en-{1}/{0}.en{1}.src".format(set, language))
  mt = path.join(resources_dir, "en-{1}/{0}.en{1}.mt".format(set, language))
  scores = path.join(resources_dir, "en-{1}/{0}.en{1}.scores".format(set, language))
  with open(source, "r", encoding='utf-8') as source, open(mt, "r", encoding='utf-8') as mt:#, open(scores, "r", encoding='utf-8') as scores:
    if path.exists(scores):
      with open(scores, "r", encoding='utf-8') as scores:
        return list(zip(source.readlines(), mt.readlines(), [float(i) for i in scores.readlines()]))
    return list(zip(source.readlines(), mt.readlines()))

class BertAsAServiceEmbedder:
  _process = None
  _imported = False

  def __init__(self):
    if not BertAsAServiceEmbedder._imported:
      from bert_serving.client import BertClient
      BertAsAServiceEmbedder._imported = True

    if BertAsAServiceEmbedder._process is None:
      BertAsAServiceEmbedder._process = subprocess.Popen(["bert-serving-start", "-model_dir=./bert/uncased_L-12_H-768_A-12", "-num_worker=4", "-max_seq_len=100"])
      atexit.register(BertAsAServiceEmbedder.kill)
    self.client = BertClient()

  def embed(self, sentences):
      out = self.client.encode(sentences)
      return out

  @staticmethod
  def kill():
    print("KILLING SERVICE")
    subprocess.call(["bert-serving-terminate", "-port=5555"])
    print("Killing Process")
    BertAsAServiceEmbedder._process.kill()

class FastTextEmbedder:
  _loaded = {}
  _imported = False
  _imported_jeiba = False

  def __init__(self, language):
    if not FastTextEmbedder._imported:
      import fastText_multilingual.fasttext as ft
      FastTextEmbedder._imported = True

    if not FastTextEmbedder._imported_jeiba and language == "zh":
      import jieba
      FastTextEmbedder._imported_jeiba = True
    embedder = "wiki.{}.align.vec".format(language)
    if embedder not in FastTextEmbedder._loaded:
      FastTextEmbedder._loaded[embedder] = ft.FastVector(vector_file=path.join(resources_dir, "FastText", embedder))
    self.ft = FastTextEmbedder._loaded[embedder]
    self.language = language
    self.unknowns = 0
    self.unknown_vocab = set()
    self.vocab = set()
    self.sentences = 0
    self.tokens = 0

  def embed(self, sentence):
    if self.language is "en":
      tokens = tokenizer.word_tokenize(sentence)
    elif self.language is "zh":
      tokens = [t for t, s, e in jieba.tokenize(sentence) if t != ' ']
    else:
      raise UnsupportedLanguageError("FastText - " + self.language)

    out = []
    for t in tokens:
      if t.lower() in self.ft:
        out.append(self.ft[t.lower()])
      else:
        print("UNKNOWN {}".format(t))
        self.unknowns += 1
        self.unknown_vocab.add(t)

    self.sentences += 1
    self.tokens += len(tokens)
    self.vocab.update(tokens)
    return out

  def stats(self):
    return {"sentences_embedded": self.sentences,
            "tokens_seen": self.tokens,
            "vocab_size_seen": len(self.vocab),
            "unknown_tokens_seen": self.unknowns,
            "unknown_vocab_seen": len(self.unknown_vocab)}


def getEmbeddedResource(modelName, embedding, language, resourceName):
  location = path.join(resources_dir, modelName, embedding, language)
  os.makedirs(location, exist_ok=True)
  resourcePath = path.join(location, resourceName)
  if path.exists(resourcePath):
    with open(resourcePath, "rb") as f:
      obj = pickle.load(f)
      return obj

  if embedding == "FastText":
    en_embed = FastTextEmbedder("en")
    b_embed = FastTextEmbedder(language)
    data = get_data(resourceName, language)
    out = []
    for tuple in data:
      e, b, s = tuple if resourceName != "test" else (*tuple, None)
      e_embedded = en_embed.embed(e)
      b_embedded = b_embed.embed(b)
      out.append((e_embedded, b_embedded) if resourceName == "test" else (e_embedded, b_embedded, s))
    print("en", en_embed.stats())
    print(language, b_embed.stats())
    with open(resourcePath, "wb") as f:
      pickle.dump(out, f)
    return out

  elif embedding == "BertAsService":
    embedder = BertAsAServiceEmbedder()
    data = get_data(resourceName, language)
    es = []
    bs = []
    ss = []
    for tuple in data:
      e, b, s = tuple if resourceName != "test" else (*tuple, None)
      es.append(e)
      bs.append(b)
      ss.append(s)
    es_array = embedder.embed(es)
    bs_array = embedder.embed(bs)
    out = []
    for i in range(es_array.shape[0]):
      if ss[0] is None:
        out.append((es_array[i], bs_array[i]))
      else:
        out.append((es_array[i], bs_array[i], ss[i]))

    with open(resourcePath, "wb") as f:
      pickle.dump(out, f)
    return out

  else:
    raise UnsupportedEmbeddingError("FastText")
