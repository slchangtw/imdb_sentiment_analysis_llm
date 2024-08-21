from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.sentiment_analysis import get_explanation_sentiment

MAX_LENGTH = 1200

if __name__ == "__main__":
    data_folder = Path("data")
    reviews = pd.read_csv(data_folder / "train.csv")

    sentiments_list = []
    for review in tqdm(reviews["review"]):
        sentiments_list.append(get_explanation_sentiment(review[:MAX_LENGTH]))

    sentiments = pd.DataFrame.from_records([s.model_dump() for s in sentiments_list])
    sentiments.rename(columns={"sentiment": "sentiment_pred"}, inplace=True)

    reviews = reviews.merge(sentiments, left_index=True, right_index=True)

    print(
        f"Accuracy: {(reviews['sentiment_x'] == reviews['sentiment_pred']).sum() / reviews.shape[0]}"
    )
    reviews.to_csv(
        data_folder
        / f"sentiments_pred_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv",
        index=False,
    )
