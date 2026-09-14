import os
import numpy as np
import librosa

# =========================================================
# Settings
# =========================================================

DATASET_PATH = "dataset/RAVDESS"
OUTPUT_PATH = "dataset/processed_mfcc"

SAMPLE_RATE = 22050
DURATION = 3
MAX_SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_MFCC = 40

N_FFT = 2048
HOP_LENGTH = 512

# =========================================================
# RAVDESS emotion mapping
# =========================================================

emotion_map = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7   # surprised
}

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

os.makedirs(OUTPUT_PATH, exist_ok=True)

X = []
y = []
actors = []

# =========================================================
# Process all actors
# =========================================================

for actor_folder in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(
        DATASET_PATH,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    if not actor_folder.startswith("Actor_"):
        continue

    actor_id = int(
        actor_folder.split("_")[1]
    )

    print(f"Processing {actor_folder}...")

    # -----------------------------------------------------
    # Process audio files
    # -----------------------------------------------------

    for file_name in sorted(os.listdir(actor_path)):

        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(
            actor_path,
            file_name
        )

        # Load audio
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        # -------------------------------------------------
        # Make every audio file exactly 3 seconds
        # -------------------------------------------------

        if len(audio) < MAX_SAMPLES:

            audio = np.pad(
                audio,
                (0, MAX_SAMPLES - len(audio))
            )

        else:

            audio = audio[:MAX_SAMPLES]

        # -------------------------------------------------
        # 1. Mel Spectrogram
        # -------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # -------------------------------------------------
        # 2. MFCC
        # -------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # -------------------------------------------------
        # 3. Delta MFCC
        # -------------------------------------------------

        delta_mfcc = librosa.feature.delta(
            mfcc
        )

        # -------------------------------------------------
        # 4. Delta-Delta MFCC
        # -------------------------------------------------

        delta2_mfcc = librosa.feature.delta(
            mfcc,
            order=2
        )

        # -------------------------------------------------
        # All features have the same time dimension
        #
        # Mel       = 64 x 130
        # MFCC      = 40 x 130
        # Delta     = 40 x 130
        # Delta2    = 40 x 130
        #
        # Total     = 184 x 130
        # -------------------------------------------------

        combined = np.concatenate(
            [
                mel_db,
                mfcc,
                delta_mfcc,
                delta2_mfcc
            ],
            axis=0
        )

        X.append(combined)

        # -------------------------------------------------
        # Extract emotion label
        # -------------------------------------------------

        emotion_code = file_name.split("-")[2]

        y.append(
            emotion_map[emotion_code]
        )

        actors.append(actor_id)


# =========================================================
# Convert to NumPy arrays
# =========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)

actors = np.array(
    actors,
    dtype=np.int64
)


# =========================================================
# Save dataset
# =========================================================

np.save(
    os.path.join(OUTPUT_PATH, "X.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_PATH, "y.npy"),
    y
)

np.save(
    os.path.join(OUTPUT_PATH, "actors.npy"),
    actors
)


# =========================================================
# Display information
# =========================================================

print("\n========================================")
print("Dataset creation completed!")
print("========================================")

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("actors shape:", actors.shape)

print("\nFeature information:")
print("Mel spectrogram: 64")
print("MFCC: 40")
print("Delta MFCC: 40")
print("Delta-Delta MFCC: 40")
print("Total feature rows:", X.shape[1])
print("Time frames:", X.shape[2])

print("\nMinimum value:", X.min())
print("Maximum value:", X.max())

print("\nEmotion distribution:")

for i, emotion in enumerate(emotion_names):

    print(
        f"{emotion:10s}: {np.sum(y == i)}"
    )

print("\nActor distribution:")

for actor_id in range(1, 25):

    count = np.sum(actors == actor_id)

    print(
        f"Actor_{actor_id:02d}: {count}"
    )

print("\nSaved files:")

print(
    "dataset/processed_mfcc/X.npy"
)

print(
    "dataset/processed_mfcc/y.npy"
)

print(
    "dataset/processed_mfcc/actors.npy"
)