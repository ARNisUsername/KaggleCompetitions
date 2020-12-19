import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = np.array([np.array(arr).reshape(28,28,1) for arr in np.array(train.drop('label', axis=1))])/255.0
y = np.array(train['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = Sequential()
model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.1,
    shear_range=10
)
datagen.fit(X)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(datagen.flow(X, y, batch_size=64), epochs=8)

theIds = np.array([i for i in range(1,28001)])
thePred = np.array(test).reshape(test.shape[0],28,28,1)
thePred = model.predict(thePred)
preds = np.argmax(thePred, axis=1)

df = pd.DataFrame({
    "ImageId":theIds,
    "Label":preds
})
df.to_csv('image_pred_epic.csv',index=False)
