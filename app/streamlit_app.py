"""Streamlit dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.data_loader import load_jobs, load_resumes
from src.matcher import build_similarity_df, load_model, rank_candidates
from src.visualize import plot_heatmap, plot_ranking

st.set_page_config(page_title="Resume-JD Matcher", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading Transformer model...")
def get_model():
    return load_model()


def read_pdf(file) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file.read()))
    return " ".join((page.extract_text() or "") for page in reader.pages)


st.title("🤖 Resume–Job Description Matcher")
st.caption("Semantic matching with Sentence Transformers (all-MiniLM-L6-v2) and cosine similarity")

# ---------------- Sidebar: input ----------------
st.sidebar.header("1. Resumes")
res_mode = st.sidebar.radio("Resume source", ["Sample data", "Upload CSV", "Upload PDFs"])
if res_mode == "Sample data":
    resumes = load_resumes()
elif res_mode == "Upload CSV":
    f = st.sidebar.file_uploader("CSV: resume_id, candidate_name, resume_text", type="csv")
    resumes = pd.read_csv(f) if f else None
else:
    files = st.sidebar.file_uploader("Resume PDFs", type="pdf", accept_multiple_files=True)
    resumes = None
    if files:
        resumes = pd.DataFrame({
            "resume_id": [f"R{i+1:02d}" for i in range(len(files))],
            "candidate_name": [Path(f.name).stem for f in files],
            "resume_text": [read_pdf(f) for f in files],
        })

st.sidebar.header("2. Job descriptions")
job_mode = st.sidebar.radio("Job source", ["Sample data", "Upload CSV", "Paste text"])
if job_mode == "Sample data":
    jobs = load_jobs()
elif job_mode == "Upload CSV":
    f = st.sidebar.file_uploader("CSV: job_id, job_title, job_description", type="csv", key="jobs")
    jobs = pd.read_csv(f) if f else None
else:
    title = st.sidebar.text_input("Job title", "Custom Role")
    desc = st.sidebar.text_area("Job description", height=180)
    jobs = pd.DataFrame({"job_id": ["J01"], "job_title": [title],
                         "job_description": [desc]}) if desc.strip() else None

if resumes is None or jobs is None:
    st.info("Provide resumes and job descriptions in the sidebar to begin.")
    st.stop()

if not {"candidate_name", "resume_text"}.issubset(resumes.columns) or \
        not {"job_title", "job_description"}.issubset(jobs.columns):
    st.error("Uploaded files are missing required columns.")
    st.stop()

with st.expander("Preview input data"):
    st.write("**Resumes**")
    st.dataframe(resumes, use_container_width=True)
    st.write("**Job descriptions**")
    st.dataframe(jobs, use_container_width=True)

# ---------------- Matching ----------------
sim_df = build_similarity_df(resumes, jobs, get_model())

tab1, tab2 = st.tabs(["🏆 Rank candidates", "🔥 Similarity matrix"])

with tab1:
    job = st.selectbox("Select a job role", list(sim_df.columns))
    max_n = len(sim_df)
    top_n = st.slider("Show top N candidates", 1, max_n, min(5, max_n)) if max_n > 1 else 1
    ranked = rank_candidates(sim_df, job, top_n)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        st.download_button("Download ranking (CSV)", ranked.to_csv(index=False),
                           file_name="ranking.csv", mime="text/csv")
    with c2:
        st.pyplot(plot_ranking(ranked, job))

with tab2:
    st.pyplot(plot_heatmap(sim_df))
    st.download_button("Download matrix (CSV)", sim_df.round(4).to_csv(),
                       file_name="similarity_matrix.csv", mime="text/csv")

st.warning("Similarity scores show semantic relevance only. They are not a hiring "
           "probability and must not be the sole basis for recruitment decisions.")
