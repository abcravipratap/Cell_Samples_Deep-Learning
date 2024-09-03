import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model_path = 'model.h5'
model = load_model(model_path)

# Define the decoding function or dictionary
def decode_label(label):
    # Replace this dictionary with your actual decoding logic
    decode_map = {0: 'Original_Value_2', 1: 'Original_Value_4'}
    return decode_map.get(label, 'Unknown')

def predict_cell_samples(Clump, UnifSize, UnifShape, MargAdh, SingEpiSize, BareNuc, BlandChrom, NormNucl, Mit):
    array = np.array([[Clump, UnifSize, UnifShape, MargAdh, SingEpiSize, BareNuc, BlandChrom, NormNucl, Mit]])
    prediction = model.predict(array)
    return prediction

# Streamlit UI elements
Clump = st.slider("Clump", min_value=1, max_value=40)
UnifSize = st.slider("UnifSize", min_value=1, max_value=40)
UnifShape = st.slider("UnifShape", min_value=1, max_value=40)
MargAdh = st.slider("MargAdh", min_value=1, max_value=40)
SingEpiSize = st.slider("SingEpiSize", min_value=1, max_value=40)
BareNuc = st.slider("BareNuc", min_value=1, max_value=40)
BlandChrom = st.slider("BlandChrom", min_value=1, max_value=40)
NormNucl = st.slider("NormNucl", min_value=1, max_value=40)
Mit = st.slider("Mit", min_value=1, max_value=40)

if st.button("Predict"):
    model_value = predict_cell_samples(Clump, UnifSize, UnifShape, MargAdh, SingEpiSize, BareNuc, BlandChrom, NormNucl, Mit)
    st.write(f"The final value is {model_value[0][0]}")  # Adjust indexing based on your model's output shape
    
    # Assuming binary classification where the model output is either 0 or 1
    final_prediction = (model_value > 0.5).astype(int)[0][0]
    
    # Decode the prediction back to original values
    decoded_value = decode_label(final_prediction)
    st.write(f"Binary Prediction: {final_prediction}")
    st.write(f"Decoded Value: {decoded_value}")


