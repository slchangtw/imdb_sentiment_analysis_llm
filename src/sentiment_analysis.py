from openai import OpenAI
from pydantic import BaseModel

from .config import settings


class SentimentAnalysis(BaseModel):
    explanation: str
    sentiment: str


PROMPT = """
You are a binary classifier for sentiment analysis of movie reviews.
Given a text, you provide a explanation (keep it as brief as possible) and classify the sentiment as positive or negative.

Text:
"""


client = OpenAI(api_key=settings.openai_api_key)


def get_explanation_sentiment(
    review: str, prompt: str = PROMPT, model="gpt-4o-2024-08-06", max_tokens=1500
) -> SentimentAnalysis:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": review},
        ],
        max_tokens=max_tokens,
        temperature=0,
        response_format=SentimentAnalysis,
    )
    message = completion.choices[0].message
    return message.parsed if message.parsed else message.refusal
