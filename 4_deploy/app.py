import os
from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np

app = Flask(__name__)

model = keras.models.load_model('/app/data/model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image from the request
    input_image = np.array(request.json['image'])

    # Normalize the image
    input_image = input_image / 255.0

    # Reshape the image
    input_image = np.reshape(input_image, (1, 28, 28, 1))

    # Use the loaded model to make a prediction
    prediction = model.predict(input_image)

    # Convert the prediction to a label
    label = np.argmax(prediction)

    # Return the label as a JSON response
    return jsonify({'label': label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
