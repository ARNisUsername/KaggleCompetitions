import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

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

#Make the model
model = keras.Sequential()
#Add 2 convential 2d layers with 32 filters(features to be extracted from the image) and a kernel size of 5(meaning we have a 5x5 weight matrix that extracts from the image to a feature map).
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
#Add a pooling layer with 2x2 pixels to apply the feature matrix(result from Conv2d layer) to another matrix, outputting the maximum value of each 2x2 patch
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Randomly deactivate 20% of the neurons in the hidden layer to reduce overfitting and generalize the model
model.add(keras.layers.Dropout(0.2))
#Add another 2 convential 2d layers, with the same filters, kernel sizes, and activation
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
model.add(keras.layers.Conv2D(32, kernel_size=5,activation='relu',input_shape=(28,28,1),padding='same',data_format='channels_last'))
#Add another 2x2 pooling layer with the max pooling technique
model.add(keras.layers.MaxPooling2D(2,2))
#Dropout another random 20% of neurons in the hidden layer
model.add(keras.layers.Dropout(0.2))
#Flatten the input layer to a 1d array so the neural network can read
model.add(keras.layers.Flatten())
#Create 2 hidden layers, each with 128 neurons, and using the relu activation
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
#Add the last hidden layer, which has 10 neurons for the 10 possible outputs, and an activation function of softmax(output vector adds up to 1.0)
model.add(keras.layers.Dense(10,activation='softmax'))

#In each image for X, when fitting the dataset, rotate it by a range 15 degrees, shift it's width and height by a range of 15%, and zoom by a range of 10%
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.1,
    shear_range=10
)
datagen.fit(X)

#Compile the model with a sparse categorical crossentropy(resulting prediction outputs vector of all 10 neurons), and use the optimizer adam
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])

model.fit(datagen.flow(X, y, batch_size=100), epochs=5)

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
test = np.array(test)
thePred = np.array(test).reshape(test.shape[0],28,28,1)
thePred = model.predict(thePred)
preds = np.argmax(thePred, axis=1)
    
ids = np.array(theIds)

df = pd.DataFrame({
    "ImageId":ids,
    "Label":preds
})

df.to_csv('image_pred_final.csv',index=False)
