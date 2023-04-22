import streamlit as st


st.set_page_config(page_title='Model Selection', page_icon='🌱')

import model2
import model1


# Define the Streamlit app
st.title('Select your Service')

# Create a dropdown to select the model
model_name = st.selectbox("What system would you like to try ?", ["Crop Recommendation System", "Plant Disease Detector System"])
if model_name == "Crop Recommendation System":
    model = model2.app()
elif model_name == "Plant Disease Detector System":
    model = model1.appmodel1();

