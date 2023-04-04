import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
from deepface import DeepFace
import pandas as pd
import numpy as np

tab1, tab2, tab3, tab4 = st.tabs(["Face Comparison", "Face Recognition", "Face Analyziss", "Crowd Analysis"])

with tab1:
   st.header("Face Comparison")
   objs = DeepFace.analyze(img_path="face_db/img19.jpg",
                           actions=['age', 'gender', 'race', 'emotion']
                           )
   st.markdown(objs)

with tab3:
    st.header("Face Comparison")
    a_one = DeepFace.analyze(img_path="face_db/img19.jpg",
                            actions=['age', 'gender', 'race', 'emotion']
                            )
    st.markdown(a_one)
    age = a_one[0]['age']
    gender = a_one[0]['dominant_gender']
    asian = a_one[0]['race']['asian']
    indian = a_one[0]['race']['indian']
    black = a_one[0]['race']['black']
    white = a_one[0]['race']['white']
    middle_eastern = a_one[0]['race']['middle eastern']
    latino_hispanic = a_one[0]['race']['latino hispanic']
    dominant_race = a_one[0]['dominant_race']
    angry = a_one[0]['emotion']['angry']
    disgust = a_one[0]['emotion']['disgust']
    fear = a_one[0]['emotion']['fear']
    happy = a_one[0]['emotion']['happy']
    sad = a_one[0]['emotion']['sad']
    surprise = a_one[0]['emotion']['surprise']
    neutral = a_one[0]['emotion']['neutral']
    dominant_emotion = a_one[0]['dominant_emotion']


    a_one_race_chart_data = pd.DataFrame(
        [asian, indian, black, white, middle_eastern, latino_hispanic],
        columns=["Asian", "Indian", "Black", "White", "Middle Eastern", "Latino Hispanic"])
    st.bar_chart(a_one_race_chart_data)

    a_one_emotion_chart_data = pd.DataFrame(
        [angry, disgust, fear, happy, sad, surprise, neutral],
        columns=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
    st.bar_chart(a_one_emotion_chart_data)
   
