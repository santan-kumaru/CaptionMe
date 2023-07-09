import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from pickle import dump, load
from tensorflow.keras.models import Model, load_model
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import string
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model

# st.set_page_config(layout="wide")
st.title("Image Caption Generator")

# Create a file uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


def extract_features(filename, model):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
             return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_path = uploaded_file

   
    if st.button("Generate"):
        max_length = 32
        tokenizer = load(open(r"C:\Users\lenovo\Downloads\nlp_projects\Image-Caption-Generator\tokenizer.p","rb"))
        model = load_model(r'C:\Users\lenovo\Downloads\nlp_projects\Image-Caption-Generator\models\model_9.h5')
        xception_model = Xception(include_top=False, pooling="avg")

        # img_path = r"C:\Users\lenovo\Downloads\nlp_projects\Image-Caption-Generator\Flickr8k_Dataset\Flicker8k_Dataset\111537222_07e56d5a30.jpg"

        photo = extract_features(img_path, xception_model)
        img = Image.open(img_path)

        description = generate_desc(model, tokenizer, photo, max_length)
        words = description.split()

        # Remove the first and last word from the list
        modified_words = words[1:-1]

        # Join the modified words back into a string
        modified_string = ' '.join(modified_words)
        st.write(modified_string)
