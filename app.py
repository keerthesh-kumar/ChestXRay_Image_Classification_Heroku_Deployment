# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Keerthesh Kumar K S
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/KerasModel2.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print(model.summary())
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    
    print("**********This is before predict1 class************")
    preds  = (model.predict(x) > 0.5).astype("int32")
    print("**********This is before predict2 class************")
    classes_names = ['NORMAL', 'PNEUMONIA']
    print("**********This is before predict3 class************")
    pred_class = classes_names[np.argmax(preds)]
    print("**********This is before predict4 class************")
    return pred_class

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print("This is f value: ",f)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print("This is after the model predict")

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        result = str(preds)
        print("This is result: ",result)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=False)
