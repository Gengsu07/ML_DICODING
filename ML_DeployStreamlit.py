import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle


st.title('Klasifikasi Kamar Messy vs Clean')
model_name = open('messy_clean_rooms.sav','rb')
model = pickle.load(model_name)
st.subheader('Upload Gambar Kamar:')
gambar = st.file_uploader(label='Gambar',type=['png','jpg'], accept_multiple_files=False)
if gambar is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('gambar asli')
        st.image(gambar)
    
    img = image.load_img(gambar,target_size=(150, 150))
    with col2:
        st.subheader('gambar 150px')
        fig = plt.figure(figsize=(8,8))
        plt.imshow(img)
        st.pyplot(fig)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Get the class indices from the train generator
    class_indices = train_generator.class_indices

    # Get the class labels from the class indices
    class_labels = list(class_indices.keys())
    # Get the predicted classes for the images
    predicted_classes = model.predict(images, batch_size=10)
    predicted_classes = predicted_classes.argmax(axis=-1)
    # Get the class names for the predicted classes
    predicted_class_names = [class_labels[i] for i in predicted_classes]
    st.write(predicted_class_names)
    st.title('Predicted:')
    st.text(predicted_class_names)