import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
from deepface import DeepFace

tab1, tab2, tab3, tab4 = st.tabs(["Face Comparison", "Face Recognition", "Face Analyziss", "Crowd Analysis"])

with tab1:
   st.header("Face Comparison")
   objs = DeepFace.analyze(img_path="face_db/img19.jpg",
                           actions=['age', 'gender', 'race', 'emotion']
                           )
   st.markdown(display(objs))
   
