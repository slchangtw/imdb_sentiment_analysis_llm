from dataclasses import dataclass

import streamlit as st

# from src.sentiment_analysis import get_explanation_sentiment


@dataclass
class SentimentAnalysis:
    explanation: str
    sentiment: str


st.title("Movie Review Sentiment Analysis")

user_review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if user_review:
        with st.spinner("Analyzing sentiment..."):
            # sentiment = get_explanation_sentiment(user_review)
            sentiment = SentimentAnalysis(
                explanation="This is a positive review.", sentiment="positive"
            )
        st.write(f"Sentiment: {sentiment.sentiment}\n\nWhy: {sentiment.explanation}")

        # Add feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 Not Helpful"):
                st.error("We're sorry to hear that. We'll work on improving!")
    else:
        st.warning("Please enter a movie review.")

st.markdown("---")
