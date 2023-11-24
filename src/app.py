import pickle
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image

# Load the model
model_path = '../model/Lightgbm.pkl'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    print(f"File not found: {model_path}")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    # Load picture
    image_california = Image.open('california.jpg')

    # Add explanatory text and picture in the sidebar
    st.sidebar.info('This app is created to predict price home California')    
    st.sidebar.image(image_california)

    # Add title
    st.title("California Prediction House App")

    # Set up the form to fill in the required data 
    longitude = st.number_input(
        'longitude', min_value= -124.35, max_value=-114.31)
    latitude = st.number_input(
        'latitude', min_value=32.54, max_value=41.95)
    housing_median_age = st.number_input(
        'housing_median_age', min_value=1, max_value=52)
    total_bedrooms = st.number_input(
        'total_bedrooms', min_value=1, max_value=10000)
    population = st.number_input(
        'population', min_value=1, max_value=50000)
    households = st.number_input(
        'households', min_value=1, max_value=15000)
    total_rooms = st.number_input(
        'total_rooms', min_value=1, max_value=10)
    median_income = st.number_input(
        'median_income', min_value=1, max_value=4000000)
    ocean_proximity = st.selectbox('Ocean Proximity', ('INLAND', 'NEAR BAY', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'))
    
    # Convert form to data frame
    input_df = pd.DataFrame([
        {
                'longitude': longitude,
                'latitude': latitude,
                'housing_median_age': housing_median_age,
                'total_bedrooms': total_bedrooms,
                'population': population,
                'households': households,
                'total_rooms': total_rooms,
                'median_income': median_income,
                'ocean_proximity': ocean_proximity,
        }
        ]
    )
        
    # Set a variabel to store the output
    output = ""
    result = ""

    # Make a prediction 
    if st.button("Predict"):
        output = model.predict(input_df)
        result = "Your best price for finding the house in California is ${:,.2f}".format(output[0])

    # Show prediction result
    st.success(result)          

if __name__ == '__main__':
    main()