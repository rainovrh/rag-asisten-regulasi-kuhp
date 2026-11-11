#!/usr/bin/env python3
"""Generate academic-grade charts and diagrams for the Evaluation Report PDF."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration for academic paper look
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Chart 1: Retrieval Ablation Study (4 Scenarios)
# -----------------------------------------------------------------------------
def generate_ablation_chart():
    scenarios = ["Skenario A\n(Fixed BM25)", "Skenario B\n(Fixed Hybrid)", "Skenario C\n(Semantic BM25)", "Skenario D\n(Usulan Utama)"]
    
    hit1 = [56.00, 54.00, 58.00, 60.00]
    hit3 = [68.00, 68.00, 78.00, 80.00]
    hit5 = [74.00, 78.00, 82.00, 88.00]
    mrr = [61.97, 62.73, 67.33, 70.57]  # Scaled to 100 for comparison

    x = np.arange(len(scenarios))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5.2))
    
    rects1 = ax.bar(x - 1.5 * width, hit1, width, label="Hit Rate@1 (%)", color="#94a3b8", edgecolor="#475569")
    rects2 = ax.bar(x - 0.5 * width, hit3, width, label="Hit Rate@3 (%)", color="#60a5fa", edgecolor="#2563eb")
    rects3 = ax.bar(x + 0.5 * width, hit5, width, label="Hit Rate@5 (%)", color="#16a34a", edgecolor="#15803d")
    rects4 = ax.bar(x + 1.5 * width, mrr, width, label="MRR (x100)", color="#ea580c", edgecolor="#c2410c")

    # Add Target Lines for Proposed Architecture (Skenario D)
    ax.axhline(y=85.0, color="#15803d", linestyle="--", linewidth=1.2, alpha=0.8, label="Target Hit@5 (>= 85%)")
    ax.axhline(y=70.0, color="#c2410c", linestyle=":", linewidth=1.2, alpha=0.8, label="Target MRR (>= 0.70)")

    ax.set_ylabel("Persentase (%) / Nilai Relatif")
    ax.set_title("Perbandingan Kinerja Temu Kembali Dokumen (Retrieval Ablation Study)", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(40, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95, ncol=2)

    # Attach value labels
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=0)

    plt.tight_layout()
    chart_path = output_dir / "grafik_studi_ablasi_retrieval.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Chart saved to {chart_path}")


# -----------------------------------------------------------------------------
# Chart 2: End-to-End Metrics vs Target Thresholds
# -----------------------------------------------------------------------------
def generate_metrics_summary_chart():
    metrics = [
        "Hit Rate@5\n(Retrieval)",
        "MRR (x100)\n(Retrieval)",
        "Answer Relevance\n(RAGAS Generation)",
        "Citation Accuracy\n(Grounding Guardrail)",
        "Faithfulness\n(RAGAS Generation)",
    ]
    achieved = [88.00, 70.57, 89.00, 93.57, 78.00]
    targets = [85.00, 70.00, 80.00, 100.00, 85.00]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    rects_achieved = ax.bar(x - width/2, achieved, width, label="Capaian Sistem (Skenario D)", color="#2563eb", edgecolor="#1d4ed8")
    rects_target = ax.bar(x + width/2, targets, width, label="Target Matriks Skripsi", color="#cbd5e1", edgecolor="#94a3b8")

    ax.set_ylabel("Nilai Metrik (%)")
    ax.set_title("Capaian Metrik Skenario D Terhadap Target Matriks Pengujian", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(50, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95)

    for rect in rects_achieved:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects_target:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color="#475569")

    plt.tight_layout()
    chart_path = output_dir / "grafik_capaian_vs_target.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    generate_ablation_chart()
    generate_metrics_summary_chart()

