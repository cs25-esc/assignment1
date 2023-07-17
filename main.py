import pickle
import numpy as np
import streamlit as st

from PIL import Image
from tensorflow.keras.preprocessing import image

import os


import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

st.title('Classification app')

m1 = tf.keras.models.load_model('model_1.h5')

imq = st.file_uploader("upload an image")


def preprocess_images(path):
    img = image.load_img(path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

if (imq):
    im = Image.open(imq)
    st.text("uploaded image")
    st.image(im)

    with open("imq.jpg", "wb") as f:
        f.write(imq.getvalue())

    image_path = os.path.abspath("imq.jpg")

    que = preprocess_images(image_path)

    u = m1.predict(que)

    predicted_class_index = np.argmax(u)

    class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    predicted_class_label = class_labels[predicted_class_index]



    st.text("you have uploaded an image of " +  predicted_class_label)
