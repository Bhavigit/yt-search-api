import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances

def returnSearchResultIndexes(
    query: str,
    df: pl.lazyframe.frame.LazyFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Computes the top-k most relevant results from the dataframe for a given search query.
    It encodes the query, compares it with stored embeddings using Manhattan distance,
    and returns the indexes of the closest matches.

    Parameters:
    - query: The search string input from the user.
    - df: A Polars LazyFrame containing video embeddings.
    - model: A SentenceTransformer model used to encode the query.

    Returns:
    - A NumPy array of indexes representing the top-k closest results.
    """

    # Encode the query string into a sentence embedding vector
    query_embedding = model.encode(query).reshape(1, -1)

    # Convert LazyFrame to eager Polars DataFrame for computation
    df_collected = df.collect()

    # Extract embedding parts: emb1 from columns 4 to 387, emb2 from 388 onward
    emb1 = df_collected.select(df.columns[4:388]).to_numpy()
    emb2 = df_collected.select(df.columns[388:]).to_numpy()

    # Compute Manhattan distance between query embedding and both parts
    dist1 = pairwise_distances(emb1, query_embedding, metric='manhattan')
    dist2 = pairwise_distances(emb2, query_embedding, metric='manhattan')

    # Sum both distances to get a combined similarity score
    dist_arr = dist1 + dist2

    # Define a similarity threshold and number of top results to return
    threshold = 40
    top_k = 5

    # Filter results below the similarity threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()

    # Sort the filtered results by distance (ascending)
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # Return the indexes of top-k closest matches
    return idx_below_threshold[idx_sorted][:top_k]


