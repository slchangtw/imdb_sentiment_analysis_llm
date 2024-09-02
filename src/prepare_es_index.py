import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import settings

MAX_LENGTH = 512


def create_index(es_client: Elasticsearch, embedding_dim: int) -> None:
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "review": {"type": "text"},
                "sentiment": {"type": "text"},
                "score": {"type": "float"},
                "id": {"type": "keyword"},
                "review_vector": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }
    es_client.indices.delete(index=settings.index_name, ignore_unavailable=True)
    es_client.indices.create(index=settings.index_name, body=index_settings)


def index_review(
    es_client: Elasticsearch,
    reviews: pd.DataFrame,
    sentence_transformer: SentenceTransformer,
) -> None:
    for _, row in tqdm(reviews.iterrows()):
        review = row["review"][:MAX_LENGTH]
        sentiment = row["sentiment"]
        score = row["score"]
        review_vector = sentence_transformer.encode(review)
        id = row["id"]

        doc = {
            "review": review,
            "sentiment": sentiment,
            "score": score,
            "id": id,
            "review_vector": review_vector.tolist(),
        }
        es_client.index(index=settings.index_name, document=doc)


if __name__ == "__main__":
    es_client = Elasticsearch(["http://localhost:9200"])
    sentence_transformer = SentenceTransformer(settings.sentence_transformer_model)
    embedding_dim = sentence_transformer.get_sentence_embedding_dimension()

    create_index(es_client, embedding_dim)

    reviews = pd.read_csv("data/sentiments_pred.csv")
    index_review(es_client, reviews, sentence_transformer)
