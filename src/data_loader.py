"""Helpers to load the sample datasets (or user supplied CSV files)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_resumes(path=None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "resumes.csv"
    df = pd.read_csv(path)
    required = {"resume_id", "candidate_name", "resume_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Resume file is missing columns: {sorted(missing)}")
    return df


def load_jobs(path=None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "job_descriptions.csv"
    df = pd.read_csv(path)
    required = {"job_id", "job_title", "job_description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Job file is missing columns: {sorted(missing)}")
    return df
