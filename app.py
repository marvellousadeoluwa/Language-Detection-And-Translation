import streamlit as st
import pickle
import numpy as np
import re
import scikit-learn as sklearn
from pathlib import Path

__version__ = '0.1.0'


BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f'trained_pipeline-{__version__}.pkl', 'rb') as f:
    model = pickle.load(f)
    #f'{BASE_DIR}/trained_pipeline-{__version__}.pkl'
# Model and prediction
classes = [
    'Arabic',
    'Danish',
    'Dutch',
    'English',
    'French',
    'German',
    'Greek',
    'Hindi',
    'Italian',
    'Kannada',
    'Malayalam',
    'Portugeese',
    'Russian',
    'Spanish',
    'Sweedish',
    'Tamil',
    'Turkish']

# helper function - mapping
def predict_pipeline(text):

    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)

    # we want to ignore figures and alphanumerics
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    pred = model.predict([text])

    return classes[pred[0]]

st.title('NLP System')

text = st.text_area('Text Input')

detect_btn = st.button('Detect Language')
translate_btn = st.button('Translate Language')

language = predict_pipeline(text)

if detect_btn:
    #st.text(language)
    with st.container():

        st.write(language)

#st.balloons()
#st.snow()
#st.metric('MyMetric', 42, 2)