import streamlit as st
import numpy as np
import requests
import config
import pickle

crop_recommend_model = pickle.load(open('models/Croprecommend.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def crop_prediction(N, P, K, ph, rainfall, city):
    """
    Predicts the crop based on the given inputs and city weather
    :params: N, P, K, ph, rainfall, city
    :return: prediction
    """
    if weather_fetch(city) != None:
        temperature, humidity = weather_fetch(city)
        data = np.array([[N, P, K, rainfall, temperature, ph, humidity]])
        my_prediction = crop_recommend_model.predict(data)
        encoded_predict = le.inverse_transform(my_prediction)
        final_prediction = encoded_predict[0]
        return final_prediction
    else:
        return None

# Define the Streamlit app
def app():
    st.title("Crop Recommendation App")

    # Get user inputs
    N = st.number_input("Enter Nitrogen e.g: 90")
    P = st.number_input("Enter Phosphorous e.g: 42")
    K = st.number_input("Enter Pottasium e.g: 43")
    ph = st.number_input("Enter pH e.g: 6.5")
    rainfall = st.number_input("Enter Rainfall e.g: 202.98")
    city = st.text_input("Enter city name e.g Kigali, Nairobi")

    # Create a button to trigger the crop prediction
    if st.button("Predict"):
        if not N:
            st.error("Please enter a value for Nitrogen")
        if not P:
            st.error("Please enter a value for Phosphorous")
        if not K:
            st.error("Please enter a value for Pottasium")
        if not ph:
            st.error("Please enter a value for pH")
        if not rainfall:
            st.error("Please enter a value for Rainfall")
        if not city:
            st.error("Please enter a value for city name")
        else:
            prediction = crop_prediction(N, P, K, ph, rainfall, city)
            if prediction is not None:
                st.success(f"The recommended crop is {prediction}")
            else:
                st.error("Invalid city name. Please try again.") 

# Run the app
if __name__ == "__main__":
    app()
