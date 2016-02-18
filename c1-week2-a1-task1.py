import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# читаем файл
with open('.\\c1-week2-a1\\sentences.txt') as f:
    lines = f.readlines()

# список слов по предложениям (строкам)
swords = list()

for l in lines:
    words = pd.Series(re.split('[^a-z]', l.lower()))
    words = words[words != '']
    swords.append(words)

# полный список слов
allwords = (pd.concat(swords)).unique()
allwords.sort()

# матрица вхождений слов в предложения
data = np.zeros((len(swords), len(allwords)), dtype=np.int)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i,j] = sum(swords[i] == allwords[j])

dist = pd.Series()
for i in range(1, data.shape[0]):
    dist.set_value(i, cosine(data[0], data[i]))

dist.sort_values(ascending=True, inplace=True)

print("Min dist indexes: ", list(dist.head(2).index))