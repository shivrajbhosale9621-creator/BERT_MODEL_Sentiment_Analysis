import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch


@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("./sentiment_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=1).item()
    return sentiment, probs.tolist()[0]


st.title("BERT Fine-Tuned Sentiment Analysis")
st.write("Enter the movie review to analyze sentiment")

user_input = st.text_area("Your text here:")

if st.button("Analyze"):
    if user_input.strip():
        sentiment, probs = predict_sentiment(user_input)
        label = "Positive" if sentiment == 1 else "Negative"
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {probs}")
    else:
        st.warning("Please enter some text.")
