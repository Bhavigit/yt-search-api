from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from app.functions import returnSearchResultIndexes

# Load a pre-trained sentence transformer model from the specified local path
model_name = 'all-MiniLM-L6-v2'
model_path = "app/data/" + model_name
model = SentenceTransformer(model_path)

# Load the video index dataset as a Polars LazyFrame for efficient querying
df = pl.scan_parquet('app/data/video-index.parquet')

# Initialize the FastAPI application
app = FastAPI()

@app.get("/")
def health_check():
    """
    Health check endpoint to confirm the API is running.
    """
    return {"health_check": "OK"}

@app.get("/info")
def info():
    """
    Returns metadata about the API.
    """
    return {
        "name": "yt-search",
        "description": "Search API for Shaw Tale-bi's YouTube videos."
    }

@app.get("/search")
def search(query: str):
    """
    Handles search requests using the provided query string.

    Parameters:
    - query: A string input used to search related YouTube videos.

    Returns:
    - A dictionary containing the titles and video IDs of the most relevant results.
    """
    # Get the indexes of top matching videos based on semantic similarity
    idx_result = returnSearchResultIndexes(query, df, model)

    # Select title and video_id for those results and return as dictionary
    return df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)
