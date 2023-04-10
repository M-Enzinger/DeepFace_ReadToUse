import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st
from deepface import DeepFace
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import tempfile



st.subheader("Face Analysis")
St.markdown("© Maximilian Enzinger, implemented using DeepFace")
st.markdown("The AI will analyse the person's face.")
st.warning("Step 1: Upload a picture")
sfa = st.file_uploader("", type=['png', 'jpg', 'img', 'jpeg'])
st.warning("Step 2: Click on 'Analyse'")
col1, col2, col3 = st.columns(3)
with col2:
    button_sfa = st.button('Analyse')
if button_sfa:
    if sfa is None:
        st.error('Upload a File First')
    else:
        st.warning("Depending on the picture, calculation can take up to 2 minutes!")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp = Path(tmp_file.name)
            fp.write_bytes(sfa.getvalue())

        a_one = DeepFace.analyze(img_path=tmp_file.name,
                                 actions=['age', 'gender', 'race', 'emotion']
                                 )

        st.success("SUCCESS. The Results:")
        if len(a_one) == 1:
            col1, col2, col3 = st.columns(3)

            im = Image.open(tmp_file.name)
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

            if len(a_one) == 1:
                s = '1 Face Found'
            else:
                s = str(len(a_one)) + ' Faces Found'

            plt.text(40, 80, s, color='blue', bbox=dict(fill=False, edgecolor='blue', linewidth=2))
            plt.show()
            st.pyplot(fig)

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

            st.subheader("Age:")
            st.info('The Person is approximately ' + str(age) + ' years old.')

            st.subheader("Gender:")
            st.info('The Person is dominantly a ' + str(gender) + '.')

            st.subheader("Race:")
            st.info('The Person is dominantly ' + str(dominant_race) + '.')
            race_chart_data = pd.DataFrame({
                'Probability in %': [asian, indian, black, white, middle_eastern, latino_hispanic],
                'Race': ["Asian", "Indian", "Black", "White", "Middle Eastern", "Latino Hispanic"]
            })
            race_chart = alt.Chart(race_chart_data).mark_bar().encode(
                y='Probability in %',
                x='Race',
            )
            st.altair_chart(race_chart, use_container_width=True)

            st.subheader("Emotion:")
            st.info('The Person is approximately ' + str(dominant_emotion) + '.')
            emotion_chart_data = pd.DataFrame({
                'Probability in %': [angry, disgust, fear, happy, sad, surprise, neutral],
                'Emotion': ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            })
            emotion_chart = alt.Chart(emotion_chart_data).mark_bar().encode(
                y='Probability in %',
                x='Emotion',
            )
            st.altair_chart(emotion_chart, use_container_width=True)

        else:
            st.error("On the provided picture are " + str(
                len(a_one)) + " Persons. But only one is allowed. To analyse several faces on one picture use 'Crowd Analysis'.")





