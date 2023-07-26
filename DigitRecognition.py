# Import all necesaary libraries
import PIL
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist
from google.colab import files

%matplotlib inline

# Upload the data
(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()

# Spliting the data into training and test sets
x_train = x_train_org.reshape(60000, 784)
x_test = x_test_org.reshape(10000, 784)

# Using one-hot encoding for answers
y_train = utils.to_categorical(y_train_org, 10)
y_test = utils.to_categorical(y_test_org, 10)

# Creating a sequential model
model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Neural Network training
model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)

# Test Neural Network with my picture with number
img = image.load_img('/content/g5.png', target_size=(28, 28), color_mode='grayscale')
img = PIL.ImageChops.invert(img)
img_test = image.img_to_array(img)
img_test = img_test.reshape(1, 784)
prediction = model.predict(img_test)
prediction = np.argmax(prediction)
print(prediction)
