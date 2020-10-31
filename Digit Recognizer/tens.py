import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

train = pd.read_csv('scaled_train.csv')
test = pd.read_csv('scaled_test.csv')

train[train.columns[1:]] = MinMaxScaler().fit_transform(train[train.columns[1:]])
test[test.columns] = MinMaxScaler().fit_transform(test[test.columns])

train.to_csv('scaled_train.csv',index=False)
test.to_csv('scaled_test.csv',index=False)

train = pd.read_csv('scaled_train.csv')
test = pd.read_csv('scaled_test.csv')

X = np.array(train.drop('label', axis=1))
y = np.array(train['label'])

newX = []
for i in X:
    newX.append(i.reshape(28,28,1))
X = np.array(newX)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=2)

def score(X_test, y_test, amount):
    score = 0
    total = amount
    theMin = random.randint(0, len(y_test)-amount-1)
    theMax = theMin + 300
    counter = 0
    for i in range(theMin, theMax):
        predictionArr = list(model.predict(X_test[i:i+1])[0])
        actualInt = predictionArr.index(max(predictionArr))
        if actualInt == y_test[i]:
            score += 1
        counter += 1
        print(f"{counter}/{total} complete")
    testAcc = score/total
    print(f"TEST ACC: {testAcc}")


theIds = [i for i in range(1,28001)]
thePred = []
for i in range(28000):
    firstPred = np.array(test.iloc[i]).reshape(1,28,28,1)
    predictionArr = list(model.predict(firstPred)[0])
    actualInt = predictionArr.index(max(predictionArr))
    thePred.append(actualInt)
    theI = i + 1
    print(f"{theI}/28000 done smh")
    
ids = np.array(theIds)
preds = np.array(thePred)

df = pd.DataFrame({
    "ImageId":ids,
    "Label":preds
})

df.to_csv('image_pred_4.csv',index=False)
