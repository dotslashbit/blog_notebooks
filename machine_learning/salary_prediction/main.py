import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder


# Load the necessary data for location, company name, and job title
data = pd.read_csv('Software_Professional_Salaries.csv')
categorical_df = data[['Location', 'Company Name', 'Job Title']].copy()
locations = data['Location'].unique
company_names = data['Company Name'].unique
job_titles = data['Job Title'].unique

# Create a Python script
model = joblib.load('linear_regression_model.pkl')

# Preprocess function for input features
def preprocess_input(input_data):
    # Perform one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(np.array(input_data).reshape(-1, 1))
    return encoded_data

# Function to predict salary
def predict_salary(features):
    # Reshape the feature vector to match the expected input shape of your model
    feature_vector = np.array(features).reshape(1, -1)
    # Make the prediction using the preprocessed feature vector
    predicted_salary = model.predict(feature_vector)
    return predicted_salary

# Streamlit app
def main():
    # App title
    st.title("Salary Prediction App")

    # Input fields
    location = st.text_input("Enter location")
    job_title = st.text_input("Enter job title")
    company_name = st.text_input("Enter company name")

    # # Preprocess input
    # location_processed = preprocess_input(location)
    # job_title_processed = preprocess_input(job_title)
    # company_name_processed = preprocess_input(company_name)
    
    # st.write(f'{location_processed} --- {job_title_processed} --- {company_name_processed}')

    # # Predict salary
    # feature_vector = np.concatenate((location_processed, job_title_processed, company_name_processed), axis=1)
    # predicted_salary = predict_salary(feature_vector)

    # # Display prediction
    # st.header("Salary Prediction")
    # st.write(f"The predicted salary is: {predicted_salary}")

if __name__ == "__main__":
    main()