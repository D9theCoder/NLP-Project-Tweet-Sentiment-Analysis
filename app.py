import streamlit as st
import re
import pickle
import numpy as np
from keras import models  # Update import
from keras.preprocessing.sequence import pad_sequences  # Update import
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
wnl = WordNetLemmatizer()
all_stopwords = set(stopwords.words('english'))
all_stopwords = [w for w in all_stopwords if w not in ['no', 'not']]

# Load the model and tokenizer
@st.cache_resource
def load_artifacts():
    try:
        # Load model with custom_objects
        model = models.load_model('model.keras', compile=False)
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

    try:
        with open('tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        raise e
        
    return model, tokenizer

def preprocessing(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', ' ', tweet, flags=re.MULTILINE)
    
    # Remove emojis and non-ASCII characters
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    
    # Remove punctuation and special characters
    tweet = re.sub(r'[^a-zA-Z0-9\s]', ' ', tweet)
    
    # Remove numbers
    tweet = re.sub(r'\d+', ' ', tweet)
    
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    # Normalize contractions
    tweet = re.sub(r"won\'t", "would not", tweet)
    tweet = re.sub(r"im", "i am", tweet)
    tweet = re.sub(r"Im", "i am", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"don\'t", "do not", tweet)
    tweet = re.sub(r"shouldn\'t", "should not", tweet)
    tweet = re.sub(r"needn\'t", "need not", tweet)
    tweet = re.sub(r"hasn\'t", "has not", tweet)
    tweet = re.sub(r"haven\'t", "have not", tweet)
    tweet = re.sub(r"weren\'t", "were not", tweet)
    tweet = re.sub(r"mightn\'t", "might not", tweet)
    tweet = re.sub(r"didn\'t", "did not", tweet)
    tweet = re.sub(r"wasn\'t", "was not", tweet)
    tweet = re.sub(r"ain\'t", "am not", tweet)
    tweet = re.sub(r"aren\'t", "are not", tweet)
    tweet = re.sub(r"\'bout", "about", tweet)
    tweet = re.sub(r"\'til", "until", tweet)
    tweet = re.sub(r"\'till", "until", tweet)
    tweet = re.sub(r"\'cause", "because", tweet)
    tweet = re.sub(r"\'em", "them", tweet)
    tweet = re.sub(r"\'n", "and", tweet)
    tweet = re.sub(r"\'d've", "would have", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub(r'unk', ' ', tweet)
    return tweet

def stemming(tweet):
    words = tweet.split()
    lemmatized_words = [wnl.lemmatize(word) for word in words if word not in all_stopwords]
    return ' '.join(lemmatized_words)

def predict_sentiment(text):
    # Preprocess the text
    text = preprocessing(text)
    text = stemming(text)
    
    # Tokenize and pad the text
    model, tokenizer = load_artifacts()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    # Make prediction
    prediction = model.predict(padded_sequences)
    sentiment_label = np.argmax(prediction)
    
    # Map prediction to sentiment label
    sentiment_mapping = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }
    
    return sentiment_mapping[sentiment_label], prediction[0][sentiment_label]

# Streamlit UI
def main():
    st.title("Tweet Sentiment Analysis")
    st.write("This app predicts the sentiment of your tweet using a BiLSTM model.")
    
    # Text input
    tweet_text = st.text_area("Enter your tweet:", height=100)
    
    if st.button("Predict Sentiment"):
        if tweet_text.strip() == "":
            st.warning("Please enter some text!")
        elif len(tweet_text.strip().split()) < 5:
            st.warning("Please enter at least 5 words for better analysis!")
        else:
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict_sentiment(tweet_text)
                
                # Display results
                st.subheader("Results:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Sentiment:** {sentiment}")
                with col2:
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Color-coded box based on sentiment
                if sentiment == "Positive":
                    st.success(f"The tweet is {sentiment} with {confidence:.2%} confidence")
                elif sentiment == "Negative":
                    st.error(f"The tweet is {sentiment} with {confidence:.2%} confidence")
                else:
                    st.info(f"The tweet is {sentiment} with {confidence:.2%} confidence")

if __name__ == "__main__":
    main()
