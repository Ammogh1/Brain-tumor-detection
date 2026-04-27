import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os
from utils import CLASSES

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "brain_tumor_detector.h5")

@st.cache_resource(show_spinner=False)
def get_model():
    """
    Loads the Tensorflow model once completely into memory.
    The decorator st.cache_resource ensures it avoids reloading during UI state changes.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return load_model(MODEL_PATH)

def predict(img_array):
    model = get_model()
    preds = model.predict(img_array, verbose=0)
    
    max_prob = float(np.max(preds))
    pred_class_idx = int(np.argmax(preds))
    pred_class_name = CLASSES[pred_class_idx]
    
    # Probabilities
    probs = {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}
    
    is_uncertain = max_prob < 0.80
    
    return {
        "predicted_class": pred_class_name,
        "confidence": max_prob,
        "all_probabilities": probs,
        "uncertain": is_uncertain
    }
