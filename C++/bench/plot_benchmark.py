#!/usr/bin/env python3
"""Plot benchmark results. Takes a single CSV with a 'dataset' column."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
})

COLORS = {"flinn": "#b5513f", "nanoflann": "#3f6fb5"}
MARKERS = {"flinn": "o", "nanoflann": "s"}


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "library": row["library"],
                "dataset": row["dataset"],
                "dims": int(row["dims"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
                "qps": int(row["num_queries"]) / float(row["query_time_s"]),
            })
    return rows


def fmt_n(n):
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} results.csv")
        sys.exit(1)

    rows = load_csv(sys.argv[1])
    outdir = Path(sys.argv[1]).parent

    synthetic = [r for r in rows if r["dataset"] == "uniform"]
    real = [r for r in rows if r["dataset"] != "uniform"]

    if synthetic:
        dims = sorted({r["dims"] for r in synthetic})
        sizes = sorted({r["n"] for r in synthetic})

        idx = defaultdict(dict)
        for r in synthetic:
            idx[(r["library"], r["dims"], r["n"])][r["k"]] = r["qps"]

        fig, axes = plt.subplots(len(sizes), len(dims),
                                 figsize=(3.5 * len(dims), 3 * len(sizes)),
                                 squeeze=False)

        for row, n in enumerate(sizes):
            for col, d in enumerate(dims):
                ax = axes[row][col]
                for lib in ["flinn", "nanoflann"]:
                    kv = idx.get((lib, d, n), {})
                    ks = sorted(kv.keys())
                    us = [1e6 / kv[k] for k in ks]
                    ax.plot(ks, us, marker=MARKERS[lib], color=COLORS[lib],
                            label=lib, linewidth=1.5, markersize=4)

                ax.set_xscale("log", base=2)
                ax.set_xticks(ks)
                ax.set_xticklabels([str(k) for k in ks])
                ax.set_ylim(bottom=0)

                if row == 0:
                    ax.set_title(f"{d}D")
                if row == len(sizes) - 1:
                    ax.set_xlabel("k")
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(f"N={fmt_n(n)}\n\u00b5s per query \u2193")

        axes[0][-1].legend(fontsize=8)
        fig.suptitle("KNN query time \u2014 uniform random data",
                     fontweight="bold", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / "benchmark.png", bbox_inches="tight")
        print(f"Saved {outdir / 'benchmark.png'}")

    if real:
        ds_dims = {r["dataset"]: r["dims"] for r in real}
        datasets = sorted({r["dataset"] for r in real}, key=lambda d: (ds_dims[d], d))
        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.5))
        if len(datasets) == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            subset = [r for r in real if r["dataset"] == ds]
            d = subset[0]["dims"]
            n = subset[0]["n"]

            idx = defaultdict(dict)
            for r in subset:
                idx[r["library"]][r["k"]] = r["qps"]

            for lib in ["flinn", "nanoflann"]:
                kv = idx.get(lib, {})
                ks = sorted(kv.keys())
                us = [1e6 / kv[k] for k in ks]
                ax.plot(ks, us, marker=MARKERS[lib], color=COLORS[lib],
                        label=lib, linewidth=1.5, markersize=4)

            ax.set_xscale("log", base=2)
            ax.set_xlabel("k")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_title(f"{ds.replace('_', ' ')} ({d}D, {fmt_n(n)})")
            ax.set_ylim(bottom=0)

        axes[0].set_ylabel("Time per query (\u00b5s) \u2193")
        axes[-1].legend(fontsize=8)
        fig.suptitle("KNN query time \u2014 real-world data", fontweight="bold", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / "benchmark_real.png", bbox_inches="tight")
        print(f"Saved {outdir / 'benchmark_real.png'}")


if __name__ == "__main__":
    main()
