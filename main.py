import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
from deepface import DeepFace

tab1, tab2, tab3, tab4 = st.tabs(["Face Comparison", "Face Recognition", "Face Analyzis", "Crowd Analysis"])

with tab1:
   st.header("Face Comparison")

   
