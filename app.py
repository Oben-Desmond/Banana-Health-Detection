import io
from flask import Flask, request, jsonify
import requests
from io import BytesIO
from flask_cors import CORS
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

app = Flask(__name__)


CORS(app)

model = load_model('banana_classifier.h5')
# Define a function that classifies an image and returns the predicted label


def classify_image(img):
    # Load the image using the file-like object
    img = load_img(img, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.

    # Make a prediction on the image
    prediction = model.predict(img_array)

    label = {}
    # Print the predicted class label
    if prediction[0][0] < 0.5:
        label = ({'value': "bad", "prediction": float(prediction[0][0])})
    else:
        label = ({'value': 'good', "prediction": float(prediction[0][0])})
    return label

# Define a REST API endpoint that accepts image files or URLs and returns the predicted label


@app.route('/classify', methods=['POST'])
def classify():

    response = requests.get(request.form['url'])
    # img = Image.open(BytesIO(response.content)).convert('RGB')
    img_content = response.content
# Create an in-memory file-like object
    img = io.BytesIO(img_content)

    label = classify_image(img)
    return jsonify(label), 200


@app.route('/home', methods=['GET'])
def home():
    return " Welcome Home"
