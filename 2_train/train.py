import tensorflow as tf
from tensorflow import keras
import numpy as np

data = np.load('/app/data/data.npz')

train_images = data['x']
train_labels = data['y']

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.save('/app/data/model.h5')
