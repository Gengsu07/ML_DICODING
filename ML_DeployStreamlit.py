import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


st.title('Klasifikasi Kamar Messy vs Clean')
model = tf.keras.models.load_model('messy_clean_rooms')
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

    classes = model.predict(images, batch_size=10)
    st.write(classes)
    if classes==0:
        st.subheader('Klasifikasi Kamar :')
        st.write('clean')
    else:
        st.subheader('Klasifikasi Kamar :')
        st.write('messy')