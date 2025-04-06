import streamlit as st
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tempfile
import os
import pickle

# Hugging Face model URL (your .h5 file)
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/2bhavyasodhi7/transformer_comment_toxicity/resolve/main/tf_model.h5"


@st.cache_resource
def load_model_from_huggingface():
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    model_file.write(requests.get(HUGGINGFACE_MODEL_URL).content)
    model_file.close()
    return load_model(model_file.name)

@st.cache_resource
def load_tokenizer_from_huggingface():
    tokenizer_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    tokenizer_file.write(requests.get(HUGGINGFACE_TOKENIZER_URL).content)
    tokenizer_file.close()
    with open(tokenizer_file.name, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load the model and tokenizer
model = load_model_from_huggingface()
tokenizer = load_tokenizer_from_huggingface()

# Streamlit UI
st.set_page_config(page_title="Toxic Comment Checker", layout="centered")
st.title("🧪 Toxic Comment Detection")
st.write("Enter a comment below and check if it's toxic.")

user_input = st.text_area("Type your comment here...")

if st.button("Check Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=128)
        prediction = model.predict(padded)[0][0]
        if prediction > 0.5:
            st.error(f"🚨 This comment is likely **TOXIC** ({prediction:.2f})")
        else:
            st.success(f"✅ This comment is **NOT toxic** ({prediction:.2f})")
