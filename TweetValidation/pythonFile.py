import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

theTrain = pd.read_csv('theTrain.csv')
theTest = pd.read_csv('theTest.csv')

X = theTrain.drop('target', axis=1)
y = theTrain['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Filtering and tokenizing of stopwords(useless words)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train['text'])

#Make long and short documents share same info and weigh down common words
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


#Train the model using MultinomialNB and the updated X_train
clf = RandomForestClassifier(n_estimators=500).fit(X_train_tf, y_train)


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
