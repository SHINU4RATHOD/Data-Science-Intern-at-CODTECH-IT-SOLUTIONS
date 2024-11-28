import streamlit as st
import re
import pickle
from keras.models import load_model
from keras.utils import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Download nltk stopwords
nltk.download('stopwords')

# Load pre-trained model and tokenizer
model = load_model("02_Task_NLP_hate_speech_rec/model.h5")
with open('02_Task_NLP_hate_speech_rec/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define stopwords and stemmer
stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[{}]'.format(re.escape(r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join([stemmer.stem(word) for word in text])
    return text

# Streamlit app
st.title("Hate Speech Classification")
st.write("Enter a text below to check whether it contains hate/abusive content or not.")

# Input from user
user_input = st.text_area("Enter text:", placeholder="Type your message here...")

# Button to predict
if st.button("Classify"):
    if user_input.strip() != "":
        # Clean and preprocess user input
        cleaned_input = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_input])
        padded = pad_sequences(seq, maxlen=300)

        # Predict
        prediction = model.predict(padded)[0][0]

        # Display result
        if prediction < 0.5:
            st.success("Result: No Hate Speech Detected")
        else:
            st.error("Result: Hate/Abusive Speech Detected")
    else:
        st.warning("Please enter some text to classify.")

# Footer
st.sidebar.header("About")
st.sidebar.info(
    """
    This application classifies text as Hate/Abusive Speech or No Hate Speech.
    Built with [Streamlit](https://streamlit.io/).
    """
)
