import os
import librosa
import numpy as np


# --------------------------------------------------
# Settings
# --------------------------------------------------

DATASET_PATH = "dataset/RAVDESS"

SAMPLE_RATE = 22050


# --------------------------------------------------
# Find all WAV files
# --------------------------------------------------

audio_files = []

for actor_folder in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(
        DATASET_PATH,
        actor_folder
    )

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.lower().endswith(".wav"):

                audio_files.append(
                    os.path.join(
                        actor_path,
                        file
                    )
                )


print("Total audio files:", len(audio_files))


# --------------------------------------------------
# Calculate durations
# --------------------------------------------------

durations = []

for i, file_path in enumerate(audio_files):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    duration = len(audio) / sr

    durations.append(duration)


durations = np.array(durations)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\nDuration statistics:")
print("--------------------")

print(
    f"Minimum duration: {durations.min():.2f} seconds"
)

print(
    f"Maximum duration: {durations.max():.2f} seconds"
)

print(
    f"Mean duration: {durations.mean():.2f} seconds"
)

print(
    f"Median duration: {np.median(durations):.2f} seconds"
)


# --------------------------------------------------
# Count files longer than 3 seconds
# --------------------------------------------------

longer_than_3 = np.sum(
    durations > 3.0
)

shorter_than_3 = np.sum(
    durations < 3.0
)

exactly_3 = np.sum(
    np.isclose(durations, 3.0)
)


print("\nFiles relative to 3 seconds:")
print("----------------------------")

print(
    f"Longer than 3 sec : {longer_than_3}"
)

print(
    f"Shorter than 3 sec: {shorter_than_3}"
)

print(
    f"Exactly 3 sec     : {exactly_3}"
)


# --------------------------------------------------
# Percentage
# --------------------------------------------------

print("\nPercentages:")

print(
    f"Longer than 3 sec : "
    f"{100 * longer_than_3 / len(durations):.2f}%"
)

print(
    f"Shorter than 3 sec: "
    f"{100 * shorter_than_3 / len(durations):.2f}%"
)