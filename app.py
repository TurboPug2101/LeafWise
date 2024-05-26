from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

from io import BytesIO

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
#from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

MODEL_PATH = "C:\\Users\\ishan\\Desktop\\webtech\\model.h5"

model = load_model(MODEL_PATH)

#model._make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(100,100))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 100, 100, 3)
    

    
    preds = model.predict(new_arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    print("here ...")
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    print(request.files)
    img_file = request.files['file'].read()

    img_size = (224, 224)

    # Load and preprocess the image
    img = image.load_img(BytesIO(img_file), target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    # Predict the class of the input image
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
        
        
    CATEGORIES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
        'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',
        'Tomato_Bacterial_spot' ,'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite' ,'Tomato__Target_Spot',
        'Tomato__YellowLeaf__Curl_Virus', 'Tomato_mosaic_virus',
        'Tomato_healthy']
    
    return CATEGORIES[class_index]



if __name__ == '__main__':
    app.run(debug=True)