"""
Pipeline Diagram (Task 6)
Generates a clean actor-learner RLHF pipeline figure for the milestone report.
Output: outputs/figures/pipeline_diagram.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(__file__), "../outputs/figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
C_ACTOR   = "#4C9BE8"   # blue   — actor
C_BUFFER  = "#F0A500"   # amber  — replay buffer
C_LEARNER = "#3DAA6E"   # green  — learner
C_DRIFT   = "#E05C5C"   # red    — drift sources
C_ARROW   = "#444444"
C_BOX_BG  = "#F7F7F7"


def box(ax, x, y, w, h, label, sublabel=None, color="#4C9BE8", fontsize=11):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color + "22",   # transparent fill
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold", color=color)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha="center", va="center", fontsize=8.5, color="#555555", style="italic")


def arrow(ax, x0, y0, x1, y1, label=None, color=C_ARROW, lw=1.8):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14, connectionstyle="arc3,rad=0.0"),
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.04, my, label, ha="left", va="center",
                fontsize=8.5, color="#333333",
                bbox=dict(fc="white", ec="none", pad=1))


def drift_note(ax, x, y, text, color=C_DRIFT):
    ax.text(x, y, text, ha="center", va="center", fontsize=8,
            color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec=color, lw=1))


# ── Layout ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Title
ax.text(5, 5.15, "Sync-or-Sink: Actor–Learner RLHF Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

# ── Nodes ──────────────────────────────────────────────────────────────────────
# Policy store (top centre)
box(ax, 5, 4.35, 2.2, 0.55, "Policy Store", "shared weights θ", color="#888888", fontsize=10)

# Actors (left side)
box(ax, 1.7, 2.8, 2.4, 0.6,  "Actor(s)",
    "vLLM / HF inference\n(fp16 / bf16)", color=C_ACTOR)

# Replay buffer (centre)
box(ax, 5,   2.8, 2.4, 0.6,  "Replay Buffer",
    "rollouts + version tags", color=C_BUFFER)

# Learner (right side)
box(ax, 8.3, 2.8, 2.4, 0.6,  "Learner",
    "PPO / TRL  (fp32)", color=C_LEARNER)

# Dataset (bottom left)
box(ax, 1.7, 1.35, 2.2, 0.5, "Preference Data",
    "HH-RLHF / RLVR tasks", color="#777777", fontsize=9.5)

# Metrics box (bottom right)
box(ax, 8.3, 1.35, 2.2, 0.5, "Stability Metrics",
    "KL · clip_frac · entropy", color="#777777", fontsize=9.5)

# ── Arrows ─────────────────────────────────────────────────────────────────────
# Policy → Actor
arrow(ax, 3.9, 4.35,  2.9, 3.1,  label="θ (lag L)", color=C_ACTOR)
# Policy → Learner
arrow(ax, 6.1, 4.35,  7.1, 3.1,  label="θ (current)", color=C_LEARNER)

# Actor → Buffer
arrow(ax, 2.9, 2.8,   3.8, 2.8,  label="rollouts", color=C_BUFFER)
# Buffer → Learner
arrow(ax, 6.2, 2.8,   7.1, 2.8,  label="batches", color=C_LEARNER)

# Learner → Policy (update)
arrow(ax, 8.3, 3.1,   6.1, 4.2,  label="update θ", color=C_LEARNER)

# Dataset → Actor
arrow(ax, 1.7, 1.62,  1.7, 2.49, color="#777777")
# Learner → Metrics
arrow(ax, 8.3, 2.49,  8.3, 1.62, color="#777777")

# ── Drift annotations ─────────────────────────────────────────────────────────
# Staleness drift (between policy store and actor)
drift_note(ax, 2.9, 3.85,
           "① Staleness drift\nlag L ∈ {0,1,2,4} updates")

# Backend mismatch drift (between actor and learner, over buffer)
drift_note(ax, 5, 1.9,
           "② Backend mismatch\nfp16 actor vs fp32 learner\n→ Δlogp σ ≈ 0.021")

# ── Legend ─────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=C_ACTOR   + "33", edgecolor=C_ACTOR,   label="Actor (rollout generation)"),
    mpatches.Patch(facecolor=C_BUFFER  + "33", edgecolor=C_BUFFER,  label="Replay Buffer"),
    mpatches.Patch(facecolor=C_LEARNER + "33", edgecolor=C_LEARNER, label="Learner (policy update)"),
    mpatches.Patch(facecolor="#FFF0F0",         edgecolor=C_DRIFT,   label="Drift source"),
]
ax.legend(handles=handles, loc="lower left", fontsize=8.5,
          framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "pipeline_diagram.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Pipeline diagram saved to {out_path}")
