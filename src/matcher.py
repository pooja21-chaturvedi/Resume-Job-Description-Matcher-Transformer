"""Core matching logic: embeddings + cosine similarity + ranking."""
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_text

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Download (first run only) and load the pretrained Transformer."""
    return SentenceTransformer(model_name)


def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    cleaned = [clean_text(t) for t in texts]
    return model.encode(cleaned, show_progress_bar=False, convert_to_numpy=True)


def similarity_matrix(resume_emb: np.ndarray, job_emb: np.ndarray) -> np.ndarray:
    """Return matrix of shape (n_resumes, n_jobs) with cosine similarity."""
    return cosine_similarity(resume_emb, job_emb)


def build_similarity_df(resumes: pd.DataFrame, jobs: pd.DataFrame,
                        model: SentenceTransformer) -> pd.DataFrame:
    """Similarity matrix as a labelled DataFrame (rows: candidates, cols: jobs)."""
    r_emb = generate_embeddings(model, resumes["resume_text"].tolist())
    j_emb = generate_embeddings(model, jobs["job_description"].tolist())
    sim = similarity_matrix(r_emb, j_emb)
    return pd.DataFrame(sim, index=resumes["candidate_name"].values,
                        columns=jobs["job_title"].values)


def rank_candidates(sim_df: pd.DataFrame, job_title: str, top_n: int = None) -> pd.DataFrame:
    """Rank candidates for one job, best match first."""
    if job_title not in sim_df.columns:
        raise KeyError(f"Unknown job title: {job_title}")
    ranked = sim_df[job_title].sort_values(ascending=False).reset_index()
    ranked.columns = ["candidate_name", "similarity_score"]
    ranked["match_percent"] = (ranked["similarity_score"].clip(lower=0) * 100).round(2)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.head(top_n) if top_n else ranked
