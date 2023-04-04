import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
from deepface import DeepFace
import pandas as pd
import numpy as np
import altair as alt

tab1, tab2, tab3, tab4 = st.tabs(["Face Comparison", "Face Recognition", "Face Analyziss", "Crowd Analysis"])


with tab3:
    st.title("Face Analysis")
    st.header("Image:")
    a_one = DeepFace.analyze(img_path="face_db/img19.jpg",
                            actions=['age', 'gender', 'race', 'emotion']
                            )

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

    st.header("Analysis Results:")

    st.subheader("Age:")
    st.info('The Person is approximately ' + str(age) + ' years old.')

    st.subheader("Gender:")
    st.info('The Person is approximately ' + str(gender))

    st.subheader("Race:")
    st.info('The Person is approximately ' + str(dominant_race))
    race_chart_data = pd.DataFrame({
        'Probability': [asian, indian, black, white, middle_eastern, latino_hispanic],
        'Race': ["Asian", "Indian", "Black", "White", "Middle Eastern", "Latino Hispanic"]
    })
    race_chart = alt.Chart(race_chart_data).mark_bar().encode(
        y='Probability',
        x='Race',
    )
    st.altair_chart(race_chart, use_container_width=True)

    st.subheader("Emotion:")
    st.info('The Person is approximately ' + str(dominant_emotion))
    emotion_chart_data = pd.DataFrame({
        'Probability': [angry, disgust, fear, happy, sad, surprise, neutral],
        'Race': ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    })
    emotion_chart = alt.Chart(emotion_chart_data).mark_bar().encode(
        y='Probability',
        x='Emotion',
    )
    st.altair_chart(emotion_chart, use_container_width=True)
