import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import tensorflow as tf
from tensorflow import keras

train = pd.read_csv('tweettrain.csv')
test = pd.read_csv('tweettest.csv')

X = train['text']
y = train['target']

def remove_punc(the_str):
    return the_str.translate(str.maketrans('', '', string.punctuation))

def has_punc(the_str):
    each_char = ' '.join(the_str).split()
    for char in each_char:
        if char in string.punctuation:
            return True
    return False

theDict = {}
counter = 1
for word in X:
    words = set(word.split())
    for unique in words:
        if has_punc(unique):
            unique = remove_punc(unique)
        if unique not in theDict.keys():
            theDict[unique.lower()] = counter
            counter += 1
theDict['<PAD>'] = 0

newX = list(X)
newList = []
for i in range(len(newX)):
    message = newX[i]
    message_split = message.lower().split()
    new_split = []
    for word in message_split:
        if has_punc(word):
            word = remove_punc(word)
        new_split.append(theDict[word])
    newList.append(new_split)

X = np.array(newList)
y = np.array(y).reshape(-1,1)

X = keras.preprocessing.sequence.pad_sequences(X, value=theDict['<PAD>'],
                                              padding='post', maxlen=250)
first_dim = int((X.shape[0]*X.shape[1])/250)
X = np.array(X).reshape(first_dim,250,1)
X = np.asarray(X).astype(np.float32)

def convert_string(the_string):
    string_arr = []
    for word in the_string.split():
        if has_punc(word):
            word = remove_punc(word)
        if word in theDict.keys():
            string_arr.append(theDict[word.lower()])
    for i in range(250-len(string_arr)):
        string_arr.append(0)
        
    return np.asarray(string_arr).astype(np.float32).reshape(1,250,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

max_index = list(set(theDict.values()))[-1] + 1
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=max_index, output_dim=16,
                                 input_length=250))
model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(16,250)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_, y epochs=8, batch_size=200, verbose=1)

theIds = np.array(test['id'])
thePreds = []
for ID in theIds:
    the_str = str(test[test['id']==ID]['text'])
    prediction = model.predict(convert_string(the_str))[0][0]
    if int(prediction) < 0.5:
        thePreds.append(0)
    else:
        thePreds.append(1)
    print(f"ID: {ID} Prediction: {prediction}")

theIds = np.array(theIds)
thePreds = np.array(thePreds)
df = pd.DataFrame({
    "id":theIds,
    "target":thePreds
})

df.to_csv('tweet_checker1.csv',index=False)
