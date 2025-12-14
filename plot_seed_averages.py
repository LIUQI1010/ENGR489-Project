# -*- coding: utf-8 -*-
"""
plot_seed_averages.py (50 generations, seed whitelist support)

Reads per-seed logs from logs_seeds/<SQL>.sql_seed<seed>_log.csv and, for each SQL file:
- Aggregates Best_Cost by Generation (mean/min/max/count) for generations 1..MAX_GEN (default 50)
- Plots the average line
- Saves per-SQL summary CSVs and PNG plots
Also produces an overall (all SQLs combined) plot and summary.

Optional: if seeds_used.txt exists, only include those seeds (so old runs won't affect results).
"""

import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ==== Config ====
LOG_DIR = "logs_seeds"
OUT_DIR = "figs"
SUMMARY_DIR = "logs_seeds_summary"
MAX_GEN = 50  # <- fixed number of generations in your runs
USE_SEED_WHITELIST = True  # read seeds_used.txt if present

FNAME_PATTERN = re.compile(r"^(?P<sql>.+)_seed(?P<seed>\d+)_log\.csv$")


def ascii_only(s: str) -> str:
    """Drop non-ASCII characters (for safe, English-only plot titles)."""
    return s.encode("ascii", "ignore").decode("ascii") or "SQL"


def sanitize_filename(s: str) -> str:
    """Make a filesystem-safe filename fragment."""
    return re.sub(r"[^\w\-.]+", "_", s)


def load_seed_whitelist(path: str = "seeds_used.txt"):
    """Return a set of seeds from seeds_used.txt, or None if not available."""
    if not USE_SEED_WHITELIST:
        return None
    if not os.path.exists(path):
        return None
    seeds = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                seeds.add(int(line))
    return seeds or None


def load_all_logs(log_dir: str) -> pd.DataFrame:
    """
    Scan *_seed<seed>_log.csv and return a DataFrame with columns:
    SQL_File, Seed, Generation, Best_Cost
    """
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    frames = []
    for fn in os.listdir(log_dir):
        if not fn.endswith("_log.csv"):
            continue
        m = FNAME_PATTERN.match(fn)
        if not m:
            continue

        sql_file = m.group("sql")       # e.g., "q1.sql"
        seed = int(m.group("seed"))
        fpath = os.path.join(log_dir, fn)

        try:
            df = pd.read_csv(fpath, encoding="utf-8")
        except Exception:
            df = pd.read_csv(fpath)

        # Require Generation + Best_Cost
        if "Generation" not in df.columns or "Best_Cost" not in df.columns:
            print(f"[WARN] Missing required columns, skipped: {fpath}")
            continue

        if "Seed" not in df.columns:
            df["Seed"] = seed

        df["Generation"] = pd.to_numeric(df["Generation"], errors="coerce")
        df["Best_Cost"] = pd.to_numeric(df["Best_Cost"], errors="coerce")
        df = df.dropna(subset=["Generation", "Best_Cost"]).copy()
        df["Seed"] = df["Seed"].fillna(seed).astype(int)
        df["SQL_File"] = sql_file

        frames.append(df[["SQL_File", "Seed", "Generation", "Best_Cost"]])

    if not frames:
        raise RuntimeError(f"No matching logs found in {log_dir} (expected *_seed<seed>_log.csv).")

    return pd.concat(frames, ignore_index=True)


def aggregate_and_plot(data: pd.DataFrame, out_dir: str, summary_dir: str, max_gen: int) -> None:
    """
    For each SQL_File, aggregate by Generation (mean/min/max) and plot:
    - mean line
    Also output an overall plot across all SQLs.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Per-SQL plots and summaries
    for sql_file, sdf in data.groupby("SQL_File"):
        # Keep only generations 1..max_gen
        sdf = sdf[(sdf["Generation"] >= 1) & (sdf["Generation"] <= max_gen)].copy()

        agg = (
            sdf.groupby("Generation", as_index=False)["Best_Cost"]
               .agg(mean="mean", min="min", max="max", count="count")
               .sort_values("Generation")
        )

        # Save per-SQL summary CSV
        safe_csv_name = sanitize_filename(f"{sql_file}_summary.csv")
        summary_path = os.path.join(summary_dir, safe_csv_name)
        agg.to_csv(summary_path, index=False, encoding="utf-8")

        # Plot: average line
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(
            agg["Generation"], agg["mean"],
            marker="o", linewidth=1.5, markersize=3, label="Average Best Cost"
        )
        plt.xlabel("Generation")
        plt.ylabel("Best Cost")
        max_n = int(agg["count"].max()) if len(agg) else 0
        title_sql = ascii_only(sql_file)
        plt.title(f"{title_sql}: 1–{max_gen} Generation Average (up to {max_n} seeds per generation)")
        plt.legend()
        plt.tight_layout()

        safe_png_name = sanitize_filename(f"{sql_file}_avg_min_max.png")
        out_png = os.path.join(out_dir, safe_png_name)
        plt.savefig(out_png)
        plt.close()

        print(f"[OK] {sql_file} -> {out_png}")
        print(f"[OK] {sql_file} -> {summary_path}")

    # Overall (all SQLs combined)
    data_all = data[(data["Generation"] >= 1) & (data["Generation"] <= max_gen)].copy()
    all_agg = (
        data_all.groupby("Generation", as_index=False)["Best_Cost"]
            .agg(mean="mean", min="min", max="max", count="count")
            .sort_values("Generation")
    )
    all_summary = os.path.join(summary_dir, "_ALL_SQL_summary.csv")
    all_agg.to_csv(all_summary, index=False, encoding="utf-8")

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(
        all_agg["Generation"], all_agg["mean"],
        marker="o", linewidth=1.5, markersize=3, label="Average Best Cost (All SQLs)"
    )
    plt.xlabel("Generation")
    plt.ylabel("Best Cost")
    plt.title(f"All SQLs: 1–{max_gen} Generation Average")
    plt.legend()
    plt.tight_layout()
    all_png = os.path.join(out_dir, "_ALL_SQL_avg_min_max.png")
    plt.savefig(all_png)
    plt.close()

    print(f"[OK] ALL -> {all_png}")
    print(f"[OK] ALL -> {all_summary}")


def main():
    # Load logs
    try:
        data = load_all_logs(LOG_DIR)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Optional: filter to seeds_used.txt (this keeps only the current run)
    seeds = load_seed_whitelist()
    if seeds is not None:
        before = len(data)
        data = data[data["Seed"].isin(seeds)].copy()
        print(f"[INFO] Filtered by seeds_used.txt: {before} -> {len(data)} rows; seeds={len(seeds)}")

    aggregate_and_plot(data, OUT_DIR, SUMMARY_DIR, MAX_GEN)


if __name__ == "__main__":
    main()
