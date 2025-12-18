import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from textblob import TextBlob

# --- NLTK Setup (Fixing the LookupError) ---
nltk.download('punkt')
nltk.download('punkt_tab')   # <-- Ye zaroori hai naye NLTK version ke liye
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# --- Load Data & Model (Cache taaki fast chale) ---
@st.cache_resource
def load_resources():
    try:
        intents = json.loads(open('intents.json').read())
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        model = load_model('chatbot_model.h5')
        return intents, words, classes, model
    except Exception as e:
        st.error(f"Error loading files: {e}. Make sure model files exist.")
        return None, None, None, None

intents, words, classes, model = load_resources()

# --- Helper Functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Customer Support Chatbot", page_icon="🤖")
st.title("🤖 Customer Support Chatbot")
st.markdown("Welcome! Ask me about **Laptops, Mobiles, Return Policy** or type your **Order ID**.")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Type your message here..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Logic Decision (Number vs Text)
    # Agar user ne Number dala (Order ID)
    if prompt.isdigit():
        res = f"📦 Order #{prompt}: Out for Delivery 🛵"
        mood_emoji = "😐" # Number ka koi mood nahi hota
        show_sentiment = False
    
    # Agar user ne Text dala (Chat)
    else:
        # AI Prediction
        ints = predict_class(prompt)
        res = get_response(ints, intents)
        
        # Sentiment Analysis (Naya Feature)
        blob = TextBlob(prompt)
        sentiment = blob.sentiment.polarity
        
        mood_emoji = "😐 (Neutral)"
        if sentiment > 0.3:
            mood_emoji = "😊 (Positive)"
        elif sentiment < -0.3:
            mood_emoji = "😡 (Negative)"
        
        show_sentiment = True

    # 3. Display Bot Response
    with st.chat_message("assistant"):
        st.markdown(res)
        # Sirf tab mood dikhayein jab Text ho (Number nahi)
        if show_sentiment:
            st.caption(f"Detected User Sentiment: {mood_emoji}")
            
    st.session_state.messages.append({"role": "assistant", "content": res})