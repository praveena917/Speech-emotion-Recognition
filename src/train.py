import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "dataset/processed"
MODEL_PATH = "models/emotion_cnn_residual_best.pth"
HISTORY_PATH = "models/residual_training_history.json"
GRAPH_PATH = "results/residual_training_curves.png"

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# 2. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))

X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))

print("Training:", X_train.shape)
print("Validation:", X_val.shape)


# ============================================================
# 4. NORMALIZATION
#    IMPORTANT: use TRAINING statistics only
# ============================================================

mean = X_train.mean()
std = X_train.std()

print("\nTraining mean:", mean)
print("Training std:", std)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std


# ============================================================
# 5. CONVERT TO PYTORCH FORMAT
# ============================================================

# Original:
# (samples, mel_bins, time)

# Required by CNN:
# (samples, channels, mel_bins, time)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

print("\nFinal training input:", X_train.shape)
print("Final validation input:", X_val.shape)


# ============================================================
# 6. CREATE DATA LOADERS
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train.numpy())

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.numpy()
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:", class_weights)


# ============================================================
# 8. RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                ),
                nn.BatchNorm2d(out_channels)
            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()


    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out


# ============================================================
# 9. RESIDUAL CNN MODEL
# ============================================================

class EmotionCNN(nn.Module):

    def __init__(self, num_classes=8):

        super().__init__()

        self.block1 = ResidualBlock(
            1,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.block2 = ResidualBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.block3 = ResidualBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        self.block4 = ResidualBlock(
            128,
            256
        )

        self.dropout = nn.Dropout(0.35)

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)

        x = self.dropout(x)

        x = self.global_pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x


# ============================================================
# 10. CREATE MODEL
# ============================================================

model = EmotionCNN(
    num_classes=8
).to(DEVICE)

print("\nModel created.")


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 13. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)


# ============================================================
# 14. TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],

    "val_loss": [],

    "train_accuracy": [],

    "val_accuracy": []
}


best_val_accuracy = 0.0
patience_counter = 0


# ============================================================
# 15. TRAINING LOOP
# ============================================================

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * inputs.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)


    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            val_total += labels.size(0)


    val_loss = val_running_loss / val_total

    val_accuracy = (
        100.0 *
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    marker = ""

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        marker = " ★ BEST"

    else:

        patience_counter += 1


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"LR: {current_lr:.6f}"
        f"{marker}"
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch "
            f"{epoch + 1}"
        )

        break


# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# 17. GENERATE ACCURACY + LOSS GRAPH
# ============================================================

epochs_completed = range(
    1,
    len(history["train_loss"]) + 1
)


plt.figure(figsize=(12, 5))


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

plt.subplot(1, 2, 1)

plt.plot(
    epochs_completed,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_completed,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

plt.subplot(1, 2, 2)

plt.plot(
    epochs_completed,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs_completed,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)


plt.tight_layout()

plt.savefig(
    GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETED")

print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best model saved to: "
    f"{MODEL_PATH}"
)

print(
    f"Training history saved to: "
    f"{HISTORY_PATH}"
)

print(
    f"Graphs saved to: "
    f"{GRAPH_PATH}"
)

print("=" * 60)