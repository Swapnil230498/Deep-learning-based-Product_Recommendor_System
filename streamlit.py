import streamlit 
from PIL import Image
import pickle
import os
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors


feature_list = (pickle.load(open('features.pkl','rb')))
filenames = np.array(pickle.load(open('filenames.pkl','rb')))


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])




streamlit.title('Fashion Recommender System')


def save_uploaded_file(uploaded_file):
    try:
        with open (os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0



def feature_extraction(img_path,model):
   img = image.load_img(img_path, target_size=(224,224))
   img_array = image.img_to_array(img)
   expanded_img_array = np.expand_dims(img_array, axis = 0)
   preprocessing_img = preprocess_input(expanded_img_array)
   result = model.predict(preprocessing_img).flatten() 
  
   normalized_result = result/norm(result)  
   return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors = 5, algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    distances , indices = neighbors.kneighbors([features])
    return indices

uploaded_file = streamlit.file_uploader('Choose an image')


if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file uploaded:
        display_image = Image.open(uploaded_file)
        streamlit.image(display_image)
        features = feature_extraction(os.path.join('uploads',uploaded_file.name),model)
        indices = recommend(features,feature_list)
        col1,col2,col3,col4,col5 = streamlit.columns(5)
        with col1:
            streamlit.image(filenames[indices[0][0]])
        with col2:
            streamlit.image(filenames[indices[0][1]])
        with col3:
            streamlit.image(filenames[indices[0][2]])
        with col4:
            streamlit.image(filenames[indices[0][3]])
        with col5:
            streamlit.image(filenames[indices[0][4]])
    else:
        streamlit.header('Some error has occured in the file upload.')

