import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and the model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    st.success("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"The required files (vectorizer.pkl and model.pkl) were not found. {e}")
except Exception as e:
    st.error(f"An error occurred while loading the model or vectorizer. {e}")

# Function to set background video from URL
def set_background(video_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            position: flex;
            z-index: -1;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5); /* Add overlay effect */
            z-index: -1;
        }}
        </style>
        <video autoplay muted loop id="bgvid" style="position: fixed; top: 0; left: 0; min-width: 100%; min-height: 100%; width: auto; height: auto; z-index: -1; object-fit: cover;">
            <source src="{video_url}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# Set the background video
set_background("https://videos.pexels.com/video-files/5378002/5378002-uhd_2560_1440_25fps.mp4")  # Replace with your video file URL

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)
        st.write(f"Transformed Message: {transformed_sms}")
        # Check if the transformed message is empty
        if not transformed_sms:
            st.error("The input message is empty. Please enter a valid message.")
        else:
            # 2. Vectorize the preprocessed message
            vector_input = tfidf.transform([transformed_sms])
            
            # 3. Predict the output using the trained model
            result = model.predict(vector_input)[0]
            probabilities = model.predict_proba(vector_input)[0]
            spam_prob = probabilities[1]
            ham_prob = probabilities[0]
            st.write(f"Spam Probability: {spam_prob * 100:.2f}%")
            st.write(f"Not Spam (Ham) Probability: {ham_prob * 100:.2f}%")
            st.write(f"Prediction Result: {result}")
            
            threshold = 0.5  # This can be adjusted
            if spam_prob > threshold:
                st.header("Spam")
            else:
                st.header("Not Spam")

            # Add a button to clear the input text
            if st.button('Clear'):
                input_sms = st.empty()
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
