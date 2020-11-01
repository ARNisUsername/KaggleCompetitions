import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

real = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

X = list(real['text'])
X1 = list(fake['text'])

for i in range(len(X1)):
    X.append(X1[i])

print("Completed X array")

amountReal = len(list(real['text']))
amountFake = len(list(fake['text']))

y = ["real" for i in range(amountReal)]
for i in range(amountFake):
    y.append("fake")

print("Completed y array")

X = np.array(X)
y = np.array(y)

df = pd.DataFrame({
    "text":X,
    "label":y
})

df.to_csv('theNews.csv',index=False)
print("Completed dataframe")
