import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "dataset/processed"
MODEL_PATH = "models/emotion_cnn_residual_best.pth"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]


# ============================================================
# RESIDUAL BLOCK
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

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                ),
                nn.BatchNorm2d(
                    out_channels
                )
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
# RESIDUAL CNN
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

        self.dropout = nn.Dropout(
            0.35
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.40
            ),

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
# LOAD TEST DATA
# ============================================================

print("Using device:", DEVICE)

print("\nLoading test dataset...")

X_train = np.load(
    os.path.join(
        DATA_PATH,
        "X_train.npy"
    )
)

X_test = np.load(
    os.path.join(
        DATA_PATH,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_PATH,
        "y_test.npy"
    )
)


# ============================================================
# NORMALIZATION
# ============================================================

mean = X_train.mean()
std = X_train.std()

print("\nTraining mean:", mean)
print("Training std:", std)

X_test = (
    X_test - mean
) / std


# ============================================================
# PYTORCH FORMAT
# ============================================================

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).unsqueeze(1)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

print("\nTest samples:", len(y_test))

print(
    "Final test input:",
    X_test.shape
)


# ============================================================
# DATA LOADER
# ============================================================

test_dataset = TensorDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD MODEL
# ============================================================

model = EmotionCNN(
    num_classes=8
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print(
    "\nLoaded model:",
    MODEL_PATH
)


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(DEVICE)

        outputs = model(inputs)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=EMOTIONS,
        digits=2
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")

print(cm)