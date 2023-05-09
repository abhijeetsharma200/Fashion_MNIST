import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images / 255.0

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images / 255.0

output_path = os.path.join("/app/data", "data.npz")
np.savez_compressed(output_path, x=train_images, y=train_labels)
