"""Command-line entry point.

Usage (from project root):
    python -m src.main
    python -m src.main --job "NLP Engineer" --top 5
"""
import argparse

from .data_loader import ROOT, load_jobs, load_resumes
from .matcher import build_similarity_df, load_model, rank_candidates
from .visualize import plot_heatmap, plot_ranking

RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description="Resume-JD matcher")
    parser.add_argument("--resumes", help="Path to resumes CSV")
    parser.add_argument("--jobs", help="Path to job descriptions CSV")
    parser.add_argument("--job", help="Job title to rank candidates for")
    parser.add_argument("--top", type=int, default=None, help="Show only top N")
    args = parser.parse_args()

    resumes, jobs = load_resumes(args.resumes), load_jobs(args.jobs)
    print(f"Loaded {len(resumes)} resumes and {len(jobs)} job descriptions.")

    model = load_model()
    sim_df = build_similarity_df(resumes, jobs, model)

    RESULTS.mkdir(exist_ok=True)
    sim_df.round(4).to_csv(RESULTS / "similarity_matrix.csv")
    plot_heatmap(sim_df).savefig(RESULTS / "similarity_heatmap.png", dpi=150)
    print("\nSimilarity matrix:\n", sim_df.round(3))

    targets = [args.job] if args.job else list(sim_df.columns)
    for title in targets:
        ranked = rank_candidates(sim_df, title, args.top)
        safe = title.lower().replace(" ", "_")
        ranked.to_csv(RESULTS / f"ranking_{safe}.csv", index=False)
        plot_ranking(ranked, title).savefig(RESULTS / f"ranking_{safe}.png", dpi=150)
        print(f"\nRanking for {title}:\n", ranked.to_string(index=False))

    print(f"\nResults saved in: {RESULTS}")


if __name__ == "__main__":
    main()
