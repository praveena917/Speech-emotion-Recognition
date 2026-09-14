import os
import numpy as np

# =========================================================
# Load MFCC + Mel dataset
# =========================================================

DATASET_PATH = "dataset/processed_mfcc"

X = np.load(
    os.path.join(DATASET_PATH, "X.npy")
)

y = np.load(
    os.path.join(DATASET_PATH, "y.npy")
)

actors = np.load(
    os.path.join(DATASET_PATH, "actors.npy")
)

print("Original dataset:")
print("X:", X.shape)
print("y:", y.shape)
print("actors:", actors.shape)


# =========================================================
# Actor-independent split
# =========================================================

# Actors 01-16 → Training
train_mask = actors <= 16

# Actors 17-20 → Validation
val_mask = (
    (actors >= 17) &
    (actors <= 20)
)

# Actors 21-24 → Final Test
test_mask = actors >= 21


# =========================================================
# Create splits
# =========================================================

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# =========================================================
# Save splits
# =========================================================

np.save(
    os.path.join(DATASET_PATH, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(DATASET_PATH, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(DATASET_PATH, "X_val.npy"),
    X_val
)

np.save(
    os.path.join(DATASET_PATH, "y_val.npy"),
    y_val
)

np.save(
    os.path.join(DATASET_PATH, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(DATASET_PATH, "y_test.npy"),
    y_test
)


# =========================================================
# Print results
# =========================================================

print("\n========================================")
print("Split completed!")
print("========================================")

print(
    "\nTraining X_train:",
    X_train.shape
)

print(
    "Training y_train:",
    y_train.shape
)

print(
    "Validation X_val:",
    X_val.shape
)

print(
    "Validation y_val:",
    y_val.shape
)

print(
    "Test X_test:",
    X_test.shape
)

print(
    "Test y_test:",
    y_test.shape
)

print("\nActor split:")

print("Training actors: 01-16")
print("Validation actors: 17-20")
print("Test actors: 21-24")

print(
    "\nTotal samples:",
    len(X_train) + len(X_val) + len(X_test)
)

print("\nSaved files:")
print("dataset/processed_mfcc/X_train.npy")
print("dataset/processed_mfcc/y_train.npy")
print("dataset/processed_mfcc/X_val.npy")
print("dataset/processed_mfcc/y_val.npy")
print("dataset/processed_mfcc/X_test.npy")
print("dataset/processed_mfcc/y_test.npy")