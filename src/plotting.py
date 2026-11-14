"""Publication-grade figure generation for the shoplifting detection benchmark."""
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_all_figures(tables_dir: Union[str, Path], figures_dir: Union[str, Path], paper_dir: Union[str, Path] = None) -> None:
    """Render Figures 1, 2, and 3 in both vector PDF and 300-DPI PNG formats."""
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_split = pd.read_csv(tables_dir / "split_rule_audit.csv")
    df_lodo = pd.read_csv(tables_dir / "lodo_benchmark_results.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
    })

    # Figure 1: In-domain split rules comparison
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    domains = ["Shop-Dataset", "MNNIT-S3", "DCSASS"]
    y_pos = np.arange(len(domains))
    for idx, dom in enumerate(domains):
        sub = df_split[df_split["domain"] == dom].set_index("split_rule")
        r_val = float(sub.loc["random_clip", "auroc"])
        s_val = float(sub.loc["strict_scene", "auroc"])
        ax.plot([s_val, r_val], [idx, idx], color="#8c969e", lw=2.4, zorder=1)
        ax.scatter([r_val], [idx], color="#c0392b", s=55, zorder=3, label="Random clips" if idx == 0 else "")
        ax.scatter([s_val], [idx], color="#1f618d", marker="D", s=55, zorder=3, label="Strict scenes" if idx == 0 else "")
        if "nominal_group" in sub.index and pd.notna(sub.loc["nominal_group", "auroc"]):
            n_val = float(sub.loc["nominal_group", "auroc"])
            ax.scatter([n_val], [idx], color="#e67e22", marker="s", s=45, zorder=2, label="Nominal groups" if idx == 1 else "")
        ax.text(r_val + 0.008, idx, f"{r_val:.3f}", va="center", ha="left", fontsize=8.5, color="#c0392b", fontweight="bold")
        ax.text(s_val - 0.008, idx, f"{s_val:.3f}", va="center", ha="right", fontsize=8.5, color="#1f618d", fontweight="bold")

    ax.axvline(0.5, color="#555555", ls=":", lw=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(domains, fontweight="bold")
    ax.set_xlim(0.40, 1.08)
    ax.set_xlabel("In-domain 5-fold AUROC (CLIP ViT-B/16 features, trained MIL head)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(figures_dir / f"fig_split_rules.{ext}", bbox_inches="tight")
    if paper_dir is not None:
        fig.savefig(Path(paper_dir) / "fig_split_rules.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Macro LODO AUROC forest plot
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    df_sorted = df_lodo.sort_values("macro4", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    for idx, r in df_sorted.iterrows():
        is_zs = str(r["model"]).startswith("zs_")
        color = "#1f618d" if is_zs else "#c0392b"
        marker = "o" if is_zs else "s"
        ax.plot([r["macro4_lo"], r["macro4_hi"]], [idx, idx], color=color, lw=1.8, alpha=0.85)
        ax.scatter([r["macro4"]], [idx], color=color, marker=marker, s=48, zorder=3)
        ax.text(r["macro4_hi"] + 0.006, idx, f"{r['macro4']:.3f} [{r['macro4_lo']:.3f}, {r['macro4_hi']:.3f}]", va="center", ha="left", fontsize=7.8)

    ax.axvline(0.5, color="#555555", ls=":", lw=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['backbone']} | {r['model']}" for _, r in df_sorted.iterrows()], fontsize=8.2)
    ax.set_xlim(0.40, 0.90)
    ax.set_xlabel("Leave-One-Dataset-Out (LODO) Macro AUROC (with 95% scene-cluster bootstrap CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(figures_dir / f"fig_macro_lodo.{ext}", bbox_inches="tight")
    if paper_dir is not None:
        fig.savefig(Path(paper_dir) / "fig_macro_lodo.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Per-dataset LODO comparison
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    corpora = ["DCSASS", "MNNIT-S3", "Shop-Dataset", "UCF-Crime"]
    x = np.arange(len(corpora))
    width = 0.25
    sel = [
        ("clip_b16", "zs_ens8_mean", "CLIP-B/16 Zero-Shot (8-prompt mean)", "#1f618d"),
        ("siglip_b16", "zs_single_max", "SigLIP-B/16 Zero-Shot (2-prompt max)", "#2e8b57"),
        ("clip_b16", "trained_head", "CLIP-B/16 Supervised MIL Head", "#c0392b"),
    ]
    for i, (bb, mod, lbl, col) in enumerate(sel):
        row = df_lodo[(df_lodo["backbone"] == bb) & (df_lodo["model"] == mod)].iloc[0]
        vals = [float(row[c]) for c in corpora]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=lbl, color=col, alpha=0.92)
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2.0, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.5, color="#555555", ls=":", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora, fontweight="bold")
    ax.set_ylim(0.35, 1.00)
    ax.set_ylabel("Held-out LODO AUROC")
    ax.legend(loc="upper left", frameon=False, fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(figures_dir / f"fig_per_dataset_lodo.{ext}", bbox_inches="tight")
    if paper_dir is not None:
        fig.savefig(Path(paper_dir) / "fig_per_dataset_lodo.pdf", bbox_inches="tight")
    plt.close(fig)
