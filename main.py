import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
from deepface import DeepFace
import pandas as pd
import numpy as np
import altair as alt

tab1, tab2, tab3, tab4 = st.tabs(["Face Comparison", "Face Recognition", "Single Face Analysis", "Crowd/ Multiple Face Analysis"])


with tab3:
    st.header("Image:")
    path = 'face_db/img19.jpg'
    a_one = DeepFace.analyze(img_path=path,
                            actions=['age', 'gender', 'race', 'emotion']
                            )

    if (len(a_one) == 1):
        col1, col2, col3 = st.columns(3)
        with col1:
            im = Image.open(path)
            # Create figure and axes
            fig, ax = plt.subplots()
            # Display the image
            ax.imshow(im)
            # Create a Rectangle patch
            for n in a_one:
                x = n['region']['x']
                y = n['region']['y']
                w = n['region']['w']
                h = n['region']['h']
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            if (len(a_one) == 1):
                s = '1 Face Found'
            else:
                s = str(len(a_one)) + ' Faces Found'

            plt.text(40, 80, s, color='blue', bbox=dict(fill=False, edgecolor='blue', linewidth=2))
            plt.show()
            st.pyplot(fig)

        age = a_one[0]['age']
        gender = a_one[0]['dominant_gender']
        asian = a_one[0]['race']['asian']*0.1
        indian = a_one[0]['race']['indian']*0.1
        black = a_one[0]['race']['black']*0.1
        white = a_one[0]['race']['white']*0.1
        middle_eastern = a_one[0]['race']['middle eastern']*0.1
        latino_hispanic = a_one[0]['race']['latino hispanic']*0.1
        dominant_race = a_one[0]['dominant_race']
        angry = a_one[0]['emotion']['angry']*0.1
        disgust = a_one[0]['emotion']['disgust']*0.1
        fear = a_one[0]['emotion']['fear']*0.1
        happy = a_one[0]['emotion']['happy']*0.1
        sad = a_one[0]['emotion']['sad']*0.1
        surprise = a_one[0]['emotion']['surprise']*0.1
        neutral = a_one[0]['emotion']['neutral']*0.1
        dominant_emotion = a_one[0]['dominant_emotion']

        st.header("Analysis Results:")

        st.subheader("Age:")
        st.info('The Person is approximately ' + str(age) + ' years old.')

        st.subheader("Gender:")
        st.info('The Person is dominantly a ' + str(gender) + '.')

        st.subheader("Race:")
        st.info('The Person is dominantly ' + str(dominant_race) + '.')
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
        st.info('The Person is approximately ' + str(dominant_emotion) + '.')
        emotion_chart_data = pd.DataFrame({
            'Probability': [angry, disgust, fear, happy, sad, surprise, neutral],
            'Emotion': ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        })
        emotion_chart = alt.Chart(emotion_chart_data).mark_bar().encode(
            y='Probability',
            x='Emotion',
        )
        st.altair_chart(emotion_chart, use_container_width=True)

    else:
        st.error("On the provided picture are " + str(len(a_one)) + ". But only one is allowed. To analyse several faces use 'Crowd Analysis'")
