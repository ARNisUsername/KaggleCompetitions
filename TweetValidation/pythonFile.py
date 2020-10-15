import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

theTrain = pd.read_csv('theTrain.csv')
theTest = pd.read_csv('theTest.csv')

X = theTrain.drop('target', axis=1)
y = theTrain['target']

#Filtering and tokenizing of stopwords(useless words)
count_vect = CountVectorizer(binary=True)
X_train_counts = count_vect.fit_transform(X['text'])

#Make long and short documents share same info and weigh down common words
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#Train the model using MultinomialNB and the updated X_train
clf = MultinomialNB().fit(X_train_tf, y)

theIds = []
theResults = []
for ID in range(10876):
    theX = theTest[theTest['id']==ID]['text']
    if len(theX) > 0:
        theIds.append(ID)
        prediction = clf.predict(count_vect.transform(theX))[0]
        theResults.append(prediction)

theIds = np.array(theIds)
theResults = np.array(theResults)

submission = pd.DataFrame({
    "id":theIds,
    "target":theResults
})

submission.to_csv('TweetChecker1.csv', index=False)
