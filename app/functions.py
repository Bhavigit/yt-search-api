import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances


def returnSearchResultIndexes(
    query: str,
    df: pl.lazyframe.frame.LazyFrame,
    model: SentenceTransformer,
) -> np.ndarray:

    query_embedding = model.encode(query).reshape(1, -1)


    df_collected = df.collect()
    emb1 = df_collected.select(df.columns[4:388]).to_numpy()
    emb2 = df_collected.select(df.columns[388:]).to_numpy()


    dist1 = pairwise_distances(emb1, query_embedding, metric='manhattan')
    dist2 = pairwise_distances(emb2, query_embedding, metric='manhattan')
    dist_arr = dist1 + dist2

    threshold = 40
    top_k = 5


    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    return idx_below_threshold[idx_sorted][:top_k]
    

