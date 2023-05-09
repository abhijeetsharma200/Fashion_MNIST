import tensorflow as tf
from tensorflow import keras
import numpy as np

data = np.load('/app/data/data.npz')

test_images = data['x'][:10000]
test_labels = data['y'][:10000]

model = keras.models.load_model('/app/data/model.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
