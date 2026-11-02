"""Plotting helpers (return matplotlib figures so Streamlit/Jupyter can reuse them)."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(sim_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(8, sim_df.shape[1] * 1.6),
                                    max(5, sim_df.shape[0] * 0.55)))
    sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Resume vs Job Description - Cosine Similarity")
    ax.set_xlabel("Job Description")
    ax.set_ylabel("Candidate")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_ranking(ranked: pd.DataFrame, job_title: str):
    data = ranked.sort_values("similarity_score")
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.45)))
    ax.barh(data["candidate_name"], data["similarity_score"], color="#2a7fba")
    for y, v in enumerate(data["similarity_score"]):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Cosine similarity")
    ax.set_title(f"Candidate ranking for: {job_title}")
    fig.tight_layout()
    return fig
