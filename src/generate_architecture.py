"""
Architecture diagram generator — produces outputs/architecture.png
using only matplotlib (no external diagram tools).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

TEAL = "#1D9E75"
NAVY = "#0D1B2A"
LIGHT_TEAL = "#E8F8F3"
ARROW_COLOR = "#1D9E75"

BOXES = [
    "Clinical Trial Data\n(2,000 Patients · 25 Features)",
    "Feature Engineering\n(PharmD-Informed Feature Store)",
    "Retention Prediction Engine\n(XGBoost + LightGBM + CatBoost)",
    "SHAP Explainability\n(Global & Per-Patient)",
    "Retention Intelligence Agent\n(9-Step Orchestration Pipeline)",
    "Patient Persona Classification\n(4 Clinical Archetypes)",
    "Intervention Engine\n(7 Evidence-Based Strategies)",
    "What-If Scenario Simulator\n(5 Protocol Change Presets)",
    "Business Impact Calculator\n(ROI & Net Savings)",
    "PDF Report + Trial Operations Dashboard",
]

SIDE_BOX = "Clinical Evidence\nRetrieval Engine\n(Lightweight RAG)"


def generate_architecture(output_path: Path = None) -> None:
    """
    Generate and save the system architecture diagram.

    Args:
        output_path: Destination path. Defaults to outputs/architecture.png.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "outputs" / "architecture.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(BOXES) * 1.4 + 1)
    ax.axis("off")

    # Title
    ax.text(5, len(BOXES) * 1.4 + 0.6,
            "AI-Powered Clinical Trial Retention Intelligence System",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=NAVY)
    ax.text(5, len(BOXES) * 1.4 + 0.15,
            "System Architecture | Dr. Reema Mohamed Sulthan, PharmD",
            ha="center", va="center", fontsize=9, color="#555555")

    box_w, box_h = 5.5, 0.85
    x_center = 5.0
    x_left = x_center - box_w / 2

    box_tops = []
    for i, label in enumerate(BOXES):
        y = (len(BOXES) - i) * 1.4 - 0.3
        box_tops.append(y)

        # Special styling for final box
        if i == len(BOXES) - 1:
            fc = NAVY
            tc = "white"
            fs = 9.5
        else:
            fc = TEAL
            tc = "white"
            fs = 9

        rect = mpatches.FancyBboxPatch(
            (x_left, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x_center, y, label,
                ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", multialignment="center")

        # Downward arrow (not after last box)
        if i < len(BOXES) - 1:
            ax.annotate(
                "", xy=(x_center, box_tops[i] - box_h / 2 - 0.45),
                xytext=(x_center, box_tops[i] - box_h / 2 - 0.02),
                arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=2),
            )

    # Side box — Clinical Evidence Retrieval Engine, connected to Step 5 (agent)
    agent_idx = 4
    agent_y = box_tops[agent_idx]
    side_x = 8.1
    side_w, side_h = 1.7, 1.1

    side_rect = mpatches.FancyBboxPatch(
        (side_x, agent_y - side_h / 2), side_w, side_h,
        boxstyle="round,pad=0.05",
        facecolor="#E8F8F3", edgecolor=TEAL, linewidth=2,
    )
    ax.add_patch(side_rect)
    ax.text(side_x + side_w / 2, agent_y, SIDE_BOX,
            ha="center", va="center", fontsize=7.5,
            color=NAVY, fontweight="bold", multialignment="center")

    # Horizontal arrow from side box to agent box
    ax.annotate(
        "", xy=(x_left + box_w, agent_y),
        xytext=(side_x, agent_y),
        arrowprops=dict(arrowstyle="<-", color=TEAL, lw=2),
    )

    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Architecture] Saved → {output_path}")


if __name__ == "__main__":
    generate_architecture()
