# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 07:46:31 2025

@author: la
"""
    
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a funtion

def diabetes_prediction(input_data):
  
     #changing the input_data to numpyarray
     input_data_as_numpy_array = np.asarray(input_data)

     #reshape the array as we are predicting for one instance
     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

     # Print the original reshaped data
     print("Original reshaped data:", input_data_reshaped)


     # adding the classifier
     prediction = loaded_model.predict(input_data_reshaped)
     print(prediction)

     if prediction[0] == 0:
       return 'The person is not diabetic'
     else:
       return 'The person is diabetic'
   


def main():
    
    #giving a titlt 
  st.title('Diabetes Prediction Web App')
    
# getting the imput data from the user

  Pregnancies = st.text_input('Number of pregnancies')
  Glucose = st.text_input('Glucose Level')
  Blood_Pressure = st.text_input('Blood Pressure Value')
  Skin_thickness = st.text_input('Skin tickness value')
  insulin = st.text_input('level of insulin')
  BMI = st.text_input('Bmi Value')
  Diabetesgradientfunction = st.text_input('value of diabetesgradientfunction')
  Age = st.text_input('Value of your Age')
  
#code for prediction

  diagnosis = ""
  
  if st.button('Diabetes Test Result'):
      diagnosis = diabetes_prediction([Pregnancies, Glucose, Blood_Pressure, Skin_thickness, insulin, BMI, Diabetesgradientfunction, Age])


  st.success(diagnosis)


if __name__ == "__main__":
    main()
 
  
  
  
















