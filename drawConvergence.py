#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch convergence plots for GA/GP experiments.
- Reads all CSVs under input_dir (default: logs/)
- For each file:
    * Averages Best_Cost every N generations (default N=20)
    * Saves aggregated CSV
    * Saves convergence plot as PNG (300dpi) and PDF (vector)
IEEE-friendly figure defaults:
    - Times-like fonts, small sizes suitable for 1-col IEEE figures
    - PDF text kept as text (fonttype=42)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ---------- IEEE-style defaults ----------
def set_ieee_matplotlib_style():
    plt.rcParams.update({
        # Fonts & text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
        "pdf.fonttype": 42,   # keep text as text in PDF
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",

        # Sizes: suitable for 1-column IEEE figure (~3.5in width)
        "figure.figsize": (3.5, 2.4),   # inches
        "figure.dpi": 300,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Lines/markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        # Axes
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Layout
        "savefig.bbox": "tight",
    })

# ---------- Core processing ----------
def process_file(file_path: str, output_dir: str, group_size: int, logy: bool):
    df = pd.read_csv(file_path)

    # Basic validation
    if "Generation" not in df.columns or "Best_Cost" not in df.columns:
        print(f"[Skip] {os.path.basename(file_path)}: missing 'Generation' or 'Best_Cost'")
        return

    # Ensure numeric
    df["Generation"] = pd.to_numeric(df["Generation"], errors="coerce")
    df["Best_Cost"]   = pd.to_numeric(df["Best_Cost"], errors="coerce")
    df = df.dropna(subset=["Generation", "Best_Cost"])

    if df.empty:
        print(f"[Skip] {os.path.basename(file_path)}: empty after numeric coercion")
        return

    # Group every N generations: 1..N, N+1..2N, ...
    df["_group"] = ((df["Generation"] - 1) // group_size) + 1
    agg = df.groupby("_group", as_index=False).agg(
        Avg_Cost=("Best_Cost", "mean"),
        Start_Gen=("Generation", "min"),
        End_Gen=("Generation", "max"),
        Count=("Best_Cost", "size"),
    )
    # x-axis at the end of each bin (N, 2N, ...)
    agg["Generation"] = agg["End_Gen"]

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = os.path.join(output_dir, f"{base}_agg_every{group_size}.csv")
    agg.to_csv(out_csv, index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(agg["Generation"], agg["Avg_Cost"], marker="o", label="Avg Best Cost")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Best Cost")
    ax.set_title(f"Convergence ({base})")
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best", frameon=False)

    out_png = os.path.join(output_dir, f"{base}_plot.png")
    out_pdf = os.path.join(output_dir, f"{base}_plot.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"[OK] {base} -> {out_png}, {out_pdf}, {out_csv}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Batch convergence plots (IEEE style).")
    parser.add_argument("--input_dir", "-i", default="logs", help="Folder containing CSV files.")
    parser.add_argument("--output_dir", "-o", default="plots", help="Folder to write outputs.")
    parser.add_argument("--group_size", "-g", type=int, default=20, help="Bin size in generations.")
    parser.add_argument("--logy", action="store_true", help="Use log scale on Y axis.")
    args = parser.parse_args()

    set_ieee_matplotlib_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all CSVs
    any_csv = False
    for name in sorted(os.listdir(args.input_dir)):
        if name.lower().endswith(".csv"):
            any_csv = True
            process_file(
                file_path=os.path.join(args.input_dir, name),
                output_dir=args.output_dir,
                group_size=args.group_size,
                logy=args.logy,
            )
    if not any_csv:
        print(f"[Info] No CSV files found under: {args.input_dir}")

if __name__ == "__main__":
    main()
