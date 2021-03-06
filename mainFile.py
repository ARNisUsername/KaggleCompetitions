import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

theTrain = pd.read_csv('train.csv')
theTest = pd.read_csv('test.csv')

d = {'female':0, 'male':1}
theTrain['Sex'] = theTrain['Sex'].map(d)
theTest['Sex'] = theTest['Sex'].map(d)

d = {'S': 0, 'C': 1, 'Q': 2}
theTrain['Embarked'] = theTrain['Embarked'].map(d)
theTest['Embarked'] = theTest['Embarked'].map(d)

X = theTrain[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = theTrain['Survived']

X = X.fillna(X.mean())
theTest = theTest.fillna(theTest.mean())

min_on_training = X.min(axis=0)
range_on_training = (X - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training

def best():
    theMax = 0
    theList = []
    CList = [0.01,0.1,1,10,100]
    GList = [0.01,0.1,1,10,100]
    for c in CList:
        for g in GList:
            model = SVC(C=c, gamma=g).fit(X_train_scaled, y_train)
            matrix = confusion_matrix(y_test, model.predict(X_test_scaled))
            if int(matrix[0][0]) + int(matrix[1][1]) > theMax:
                theMax = int(matrix[0][0]) + int(matrix[1][1])
                theList = [c,g]
    return theList

def scaleNum(theNum):
    return (theNum - min_on_training) / range_on_training

model = SVC(kernel='poly', C=best()[0], gamma=best()[1]).fit(scaleNum(X), y)

totalIds = []
Y_Pred = []

for theId in range(892, 1310):
    theX = theTest[theTest['PassengerId']==theId][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    totalIds.append(theId)
    Y_Pred.append(model.predict(scaleNum(theX)))

totalIds = np.array(totalIds)
Y_Pred = np.array(Y_Pred)
Y_Pred = Y_Pred.ravel()

submission = pd.DataFrame({
    "PassengerId": totalIds,
    "Survived": Y_Pred
})

submission.to_csv('theSubmission.csv', index=False)
