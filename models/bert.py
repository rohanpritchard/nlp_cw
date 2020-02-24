import pickle

import numpy as np
from bert_serving.client import BertClient

from utils.get_data import get_data
from scipy.stats.stats import pearsonr

bc = BertClient()
e = bc.encode(['Hamilton oversaw his colleagues under the elective reign of George Washington.',
               'On 29 December 2013, Bergling debuted his new track "Dreaming of Me", featuring vocals from Audra Mae, via episode 19 of his LE7ELS podcast.',
               'Wenceslaus abdicated in favor of Otto of Bavaria in 1305.'])
c = bc.encode(['汉密尔顿在乔治 · 华盛顿的选任统治下监督他的同事们.',
               '2013 年 12 月 29 日 ， 柏格林通过他的 LE7ELS 播客第 19 集 ， 推出了他的新的 "我的梦想" ， 其中有奥黛拉 · 梅的歌声。',
               '温切斯劳斯于 1305 年向巴伐利亚的奥托投降.'])

data = get_data("dev", german=False)
out = []
js = []
ss = []
print("starting looop")
for e, c, s in data:
  print(e)
  ee = bc.encode([e])
  ec = bc.encode([c])
  out.append((ee, ec, s))
  d = np.abs(np.subtract(ee, ec))
  j = np.sum(d, axis=1)[0]
  js.append(j)
  ss.append(s)

with open("bert_encoded", "wb") as file:
  pickle.dump(out, file)

print("PEARSON:", pearsonr(np.array(ss), np.array(js)))