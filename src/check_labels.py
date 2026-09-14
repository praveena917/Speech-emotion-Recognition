import numpy as np

y_train = np.load("dataset/processed/y_train.npy")

emotion_names = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

unique, counts = np.unique(y_train, return_counts=True)

print("Training class distribution:\n")

for label, count in zip(unique, counts):
    print(f"{emotion_names[label]:10s}: {count}")