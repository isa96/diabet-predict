from pathlib import Path
import sys
import joblib as job
import streamlit as st
import numpy as np
import keras

model_path = Path(__file__).parents[1] / 'code/models/best_model.h5'
scaler_path =Path(__file__).parents[1] / 'code/models/scaler'

# Load model
with open(model_path, 'rb') as m:
    diabet_model = keras.models.load_model(model_path)

# Load scaler
with open(scaler_path, 'rb') as s:
    sc = job.load(s)

# Title web
st.title('Sistem Prediksi Diabetes')

# Kolom
col1, col2 = st.columns(2)

# Buat form input fitur
with col1:
    Pregnancies = st.text_input('Input nilai Pregnancies')

with col2:
    Glucose = st.text_input('Input nilai Glucose')

with col1:
    BloodPressure = st.text_input('Input nilai BloodPressure')

with col2:
    BMI = st.text_input('Input nilai BMI')

with col1:
    DiabetesPedigreeFunction = st.text_input('Input nilai DiabetesPedigreeFunction')

with col2:
    Age = st.text_input('Input nilai Age')

# Code untuk prediksi
diabet_diagnosis = ''

# Button prediksi
if st.button('Test Prediksi Diabetes'):
    # Create an input array from user inputs
    input_data = np.array([float(Pregnancies), float(Glucose), float(BloodPressure), float(BMI), float(DiabetesPedigreeFunction), float(Age)])

    # Reshape the input data to the shape expected by the scaler (1 sample with 6 features)
    input_data = input_data.reshape(1, -1)

    # Scale the input data
    scaled_input_data = sc.transform(input_data)
    
    diabet_predict = diabet_model.predict(scaled_input_data)

    if round(diabet_predict[0][0]) < 1:
        diabet_diagnosis = 'Pasien Tidak Terkena Diabetes'
    else:
        diabet_diagnosis = 'Pasien Terkena Diabetes'

    st.success(diabet_diagnosis)
