import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# SETTINGS
# ============================================================

RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)


# ============================================================
# EMOTION NAMES
# ============================================================

emotions = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]


# ============================================================
# FINAL TEST RESULTS
# ============================================================

confusion_matrix = np.array([
    [12, 1, 3, 0, 0, 0, 0, 0],
    [6, 18, 2, 4, 0, 1, 1, 0],
    [4, 0, 18, 0, 2, 1, 0, 7],
    [5, 6, 9, 5, 2, 0, 5, 0],
    [1, 0, 5, 0, 21, 0, 1, 4],
    [0, 0, 13, 8, 2, 5, 1, 3],
    [0, 0, 4, 0, 6, 0, 22, 0],
    [0, 0, 5, 0, 3, 0, 1, 23]
])


f1_scores = np.array([
    0.55,
    0.63,
    0.40,
    0.20,
    0.62,
    0.26,
    0.70,
    0.67
])


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=emotions
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

ax.set_title(
    "Speech Emotion Recognition - Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Emotion"
)

ax.set_ylabel(
    "True Emotion"
)

plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_PATH,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    confusion_path
)


# ============================================================
# 2. F1-SCORE GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars = ax.bar(
    emotions,
    f1_scores
)

ax.set_title(
    "F1-Score by Emotion"
)

ax.set_xlabel(
    "Emotion"
)

ax.set_ylabel(
    "F1-Score"
)

ax.set_ylim(
    0,
    1
)

ax.set_xticks(
    range(len(emotions))
)

ax.set_xticklabels(
    emotions,
    rotation=30,
    ha="right"
)


# Add F1 values above bars

for bar, score in zip(
    bars,
    f1_scores
):

    ax.text(
        bar.get_x() +
        bar.get_width() / 2,
        score + 0.02,
        f"{score:.2f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

f1_path = os.path.join(
    RESULTS_PATH,
    "f1_scores.png"
)

plt.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    f1_path
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nEvaluation graphs created successfully.")

print(
    "\nConfusion Matrix:",
    confusion_path
)

print(
    "F1 Score Graph:",
    f1_path
)