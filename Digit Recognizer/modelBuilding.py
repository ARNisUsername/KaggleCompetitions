import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

#Read the csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Scale all of the columns(except for the labels column) to 0-1
train[train.columns[1:]] = MinMaxScaler().fit_transform(train[train.columns[1:]])
test[test.columns] = MinMaxScaler().fit_transform(test[test.columns])


train.to_csv('scaled_train.csv',index=False)
test.to_csv('scaled_test.csv',index=False)

#Read the new csv file(In the real code, you would jump straight to this line)
train = pd.read_csv('scaled_train.csv')
test = pd.read_csv('scaled_test.csv')

#Create the X and y, and change them to numpy arrays so you can reshape them
X = np.array(train.drop('label', axis=1))
y = np.array(train['label'])

#Reshape the X array to something the keras layers can read
newX = []
for i in X:
    newX.append(i.reshape(28,28,1))
X = np.array(newX)

#Make the model
model = keras.Sequential()
#Add a Convential 2d neural network(neural network common for image classification) with 32 filters to extract from the image data
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
model.add(keras.layers.MaxPooling2D(2,2))
#Flattens the layers to 1 dimension for the neural network to properly read
model.add(keras.layers.Flatten())
#Add 2 128 neuron hidden layers, where each neuron's function applies ReLU
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
#Add the output layer with 10 neurons(each representing a different number)
model.add(keras.layers.Dense(10,activation='softmax'))

#Compile the model with sparse categorical crossentropy(for classification tasks, outputs array of the 10 neurons)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
#Fit the model with X and y, and with 2 epochs
model.fit(X, y, epochs=2)

#A function made for finding out the Test accuracy. It takes a random batch of {amount} images and labels from X_test and y_test and asseses them. 
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

#Create a list of the Ids and the predictions
theIds = [i for i in range(1,28001)]
thePred = []
for i in range(28000):
    #Locate the row in the scaled test csv file, change it to a numpy array and reshape it to 4 dimensions so the model can predict from it
    firstPred = np.array(test.iloc[i]).reshape(1,28,28,1)
    #Get the array with info on the 10 neurons, and store the index(from 0-9, very convient) of the maximum number to actualInt, to be appended to thePred
    predictionArr = list(model.predict(firstPred)[0])
    actualInt = predictionArr.index(max(predictionArr))
    thePred.append(actualInt)
    theI = i + 1
    print(f"{theI}/28000 done")
    
#Make each of the lists into numpy arrays to be transformed into a pandas dataframe
ids = np.array(theIds)
preds = np.array(thePred)

#Create a pandas dataframe with the new predictions
df = pd.DataFrame({
    "ImageId":ids,
    "Label":preds
})

#Put the pandas dataframe to a csv file to output to Kaggle(Has a 96.5% accuracy so far)
df.to_csv('image_pred_4.csv',index=False)
