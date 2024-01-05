
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

hide_streamlit_style = """

            #MainMenu {visibility: hidden;}


            """
st.title('Brain Tumor Detection using CNN')

def main() :
    file_uploaded = st.file_uploader('Choose an image...', type = ['jpg','jpeg','png'])

    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(image)

        if st.button('Predict', type = 'primary'):

           with st.spinner('Loading the model...'):

                # load model

                model = load_model('/content/drive/MyDrive/Colab Notebooks/brain.keras')


           # Image Augmentation of validation data 

           test_datagen = ImageDataGenerator(rescale=1./255)
           test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/Dataset/test',target_size=(224,224),batch_size=16,shuffle=False,class_mode='binary')


           # Preprocess the image

           # Resize the image to 224x224
           image = image.resize((224, 224))

           # Convert grayscale image to RGB
           image = image.convert('RGB')


           test_image = tf.keras.utils.img_to_array(image)
           test_image = np.expand_dims(test_image, axis=0)


           # Predict

           with st.spinner('Just a moment...'):
                result = model.predict(test_image)
                score = model.evaluate(test_set, verbose = 0)
          
           if result[0][0] == 1:
                st.write('Yes. Brain Tumor Detected')
           else:
                st.write('NO. Brain Tumor Not Detected')

           st.write('Accuracy (%):', f"{score[1]*100:.2f}")


if __name__ == '__main__':
    main()


