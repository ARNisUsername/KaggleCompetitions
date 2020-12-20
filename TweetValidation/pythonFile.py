import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

train = pd.read_csv('train_tweet.csv')[['text', 'target']]
test = pd.read_csv('test_tweet.csv')[['text', 'id']]

def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

X = np.array([clean_text(text) for text in train['text']])
y = np.array(train['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = Pipeline([
    ("vect",CountVectorizer()),
    ("trans",TfidfTransformer()),
    ("model",SVC())
])
model.fit(X, y)

theIds = np.array(test['id'])
theText = np.array([clean_text(text) for text in test['text']])
all_preds = np.array(model.predict(theText))

df = pd.DataFrame({
    "id":theIds,
    "target":all_preds
})

df.to_csv('tweet_sub.csv', index=False)
