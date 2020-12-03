import os

import numpy as np
from flask import Flask, render_template, redirect, request, url_for
import cv2
import pickle
from keras import backend as K
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/img', methods=['POST'])
def img():
    getimg = request.form['logo']
    try:
        img = cv2.imread("/home/dolan/Documents/datasets/" + getimg)
        # print(img)
        # getimg = cv2.UMat(getimg)
        default_image_size = tuple((256, 256))
        img = cv2.resize(img, default_image_size)
    except Exception as e:
        print(str(e))
    '''
    loaded_model = pickle.load(open('model/cnn_model.pkl', 'rb'))
    imag = np.expand_dims(img, axis=0)
    k = np.array(imag)
    # pre = loaded_model.predict(k)
    pre = loaded_model.predict(k).tolist()
    K.clear_session()
    print(pre)
    
    for i in pre:
        print(i)
        if i[0] == 1.0:
            disease = "Bacterial Spot"
            return redirect(url_for('disease_name', disease=disease))
        elif i[1] == 1.0:
            disease = "Healthy"
            return redirect(url_for('disease_name', disease=disease))
        elif i[2] == 1.0:
            disease = "Late Bright"
            return redirect(url_for('disease_name', disease=disease))
        elif i[3] == 1.0:
            disease = "Tomato Mosaic"
            return redirect(url_for('disease_name', disease=disease))
        elif i[4] == 1.0:
            disease = "Yellow Curl"
            return redirect(url_for('disease_name', disease=disease))
        else:
            return redirect(url_for('index'))
    '''
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest")
    loaded_model1 = pickle.load(open('model/cnn_model1.pkl', 'rb'))
    img = aug.standardize(img)
    img = img / 255
    imag1 = np.expand_dims(img, axis=0)
    k1 = np.array(imag1)
    # pre = loaded_model.predict(k)
    pre1 = loaded_model1.predict(k1).tolist()
    # K.clear_session()
    print(pre1)
    for j in pre1:
        if j[0] < 0.5:
            disease = "Healthy"
            K.clear_session()
            return redirect(url_for('disease_name', disease=disease))
        elif j[0] >= 0.5:
            loaded_model = pickle.load(open('model/cnn_model.pkl', 'rb'))
            # imag = np.expand_dims(img, axis=0)
            # k = np.array(imag)
            # pre = loaded_model.predict(k)
            pre = loaded_model.predict(k1).tolist()
            K.clear_session()
            max = -1
            index = 0
            for i in pre:
                for a in range(0, 5):
                    if i[a] > max:
                        max = i[a]
                        index = a

            if index == 0:
                disease = "Bacterial Spot"
                return redirect(url_for('disease_name', disease=disease))
            elif index == 1:
                disease = "Healthy"
                return redirect(url_for('disease_name', disease=disease))
            elif index == 2:
                disease = "Late Bright"
                return redirect(url_for('disease_name', disease=disease))
            elif index == 3:
                disease = "Tomato Mosaic"
                return redirect(url_for('disease_name', disease=disease))
            elif index == 4:
                disease = "Yellow Curl"
                return redirect(url_for('disease_name', disease=disease))
            else:
                return redirect(url_for('index'))

    # return redirect(url_for('index'))


@app.route('/index/disease/<disease>')
def disease_name(disease):
    return render_template('index.html', disease_name=disease)


if __name__ == '__main__':
    app.run()
