import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(0)

train = pd.read_csv('titanictrain.csv')
test = pd.read_csv('titanictest.csv')

train['Age'] = train['Age'].fillna(int(np.mean(train['Age'])))
test['Age'] = test['Age'].fillna(int(np.mean(test['Age'])))

train['Cabin'] = train['Cabin'].fillna("none")
test['Cabin'] = test['Cabin'].fillna("none")

train['Embarked'] = train['Embarked'].fillna("none")
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

train = train.drop(['Name','Ticket'], axis=1)
test = test.drop(['Name','Ticket'], axis=1)

for col in test.columns:
    try:
        int(train[col].iloc[[0]])
    except:
        d = {}
        counter = 0
        for key in train[col].unique():
            d[key] = counter
            counter += 1
        train[col] = train[col].map(d)
        
        d = {}
        counter = 0
        for key in test[col].unique():
            d[key] = counter
            counter += 1
        test[col] = test[col].map(d)


X = train.drop('Survived',axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_grid = {'learning_rate':[0.01,0.1,1,10,100],'max_depth':[2,3,4,5,6]}
grid = GridSearchCV(GradientBoostingClassifier(),param_grid,verbose=1)
model = grid.fit(X_train, y_train)

theIds = np.array([i for i in range(892,1310)])
allPredictions = []
for ID in theIds:
    X = test[test['PassengerId']==ID]
    prediction = model.predict(X)
    allPredictions.append(prediction[0])
    count = ID - 891
    print(f"{count}/418 completed")
    
allPredictions = np.array(allPredictions)

df = pd.DataFrame({
    "PassengerId":theIds,
    "Survived":allPredictions
})

df.to_csv('theTitanicPred.csv',index=False)

