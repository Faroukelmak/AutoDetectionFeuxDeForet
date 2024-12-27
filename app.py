import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt


model = load_model('model2.h5')

# Fonction de prétraitement d'image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Taille de l'image
    img_array = img_to_array(img)  # Convertir l'image en tableau numpy
    img_array = img_array / 255.0  # Normalisation de l'image
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour correspondre à l'entrée du modèle
    return img_array


st.title("Prédiction d'Image : Fire ou Non Fire")

uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None : 
    
    # Afficher l'image téléchargée
    st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)
    
    
    #pretraiter l ' image  
    
    img_array = preprocess_image(uploaded_image)
    
    #predire avec le modele 
    
    prediction = model.predict(img_array)
    predicted_class = 1 if prediction[0] > 0.5 else 0  # Seuil de 0.5
    
    ## afficher la prediction 
    
    if predicted_class == 1 :
        st.write("Prédiction : **il y a de feu**")
    else : 
        st.write("Prédiction : **pas de feu**")
        
    # Afficher la prédiction brute pour référence
    st.write(f"Prédiction brute : {prediction[0][0]:.4f}")
    
    
    
    
    