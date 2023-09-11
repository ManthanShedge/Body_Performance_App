# Import required libraries
import pandas as pd
import streamlit as st
import joblib

# Pickle the model
model=joblib.load("lightgbm_model.pkl")
    
# Streamlit App
st.title("Body Performance Prediction App")

# Sidebar
st.sidebar.header("Parameters")
# Prediction
st.header("Body Performance Prediction App")
st.subheader("Enter Data for Prediction in the sidebar")
user_input = {}
user_input['age'] = st.sidebar.slider("Age", 10, 90, step=1)
user_input['height_cm'] = st.sidebar.slider("Height(in cm)", 130.0, 250.0, step=0.1)
user_input['weight_kg'] = st.sidebar.slider("Weight(in kg)", 20.0, 200.0, step=0.1)
user_input['body fat_%'] = st.sidebar.slider("Body Fat%", 1.0, 50.0, step=0.1)
user_input['diastolic'] = st.sidebar.slider("Diastolic Blood pressure", 40, 140, step=1)
user_input['gripForce'] = st.sidebar.slider("Gripforce", 20.0, 70.0, step=0.1)
user_input['sit_bend'] = st.sidebar.slider("Sit and bend (in cm)", 0.0, 40.0, step=0.1)
user_input['sit_ups'] = st.sidebar.slider("Sit ups count", 0, 100, step=1)
user_input['broad_jump'] = st.sidebar.slider("Broad jump(in cm)", 70, 250, step=1)
gender = st.sidebar.selectbox(f"Select Gender", ['Male','Female'])
if gender =='Female':
    user_input['M']=0
else:
    user_input['M']=1
# Convert user input to DataFrame

user_df = pd.DataFrame([user_input])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(user_df)
    st.subheader("Prediction Result")
    if prediction[0]==0:
        st.write(f"You have an excellent body performance!")
    elif prediction[0]==1:
        st.write(f"You have a good body performance")
    elif prediction[0]==2:
        st.write(f"You have an average body performance and improve")
    else:
        st.write(f"You have a bad body performance and need to improve a lot")

