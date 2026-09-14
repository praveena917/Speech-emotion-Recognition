# Speech Emotion Recognition Using Deep Learning

A deep learning project that classifies human speech into **8 different emotions** using **Mel-spectrogram features** and a **Residual Convolutional Neural Network (CNN)**.

**Technologies:** Python · PyTorch · Librosa · NumPy · Scikit-learn · Matplotlib

---

## 📌 Project Overview

Speech Emotion Recognition (SER) is a deep learning task that aims to automatically identify the emotional state expressed in human speech.

In this project, audio recordings from the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset are processed into Mel-spectrograms and used to train a **Residual CNN** for emotion classification.

A **speaker-independent train, validation, and test split** is used so that speakers present in the test set are never seen during training.

This provides a more realistic evaluation of how well the model generalizes to unseen speakers.

### Dataset

| Property | Details |
|---|---|
| Dataset | RAVDESS |
| Total Audio Files | 1,440 |
| Number of Actors | 24 |
| Number of Classes | 8 |
| Sampling Rate | 22,050 Hz |
| Audio Duration Used | 3 seconds |
| Feature | Mel-Spectrogram |
| Feature Shape | 64 × 130 |

### Final Performance

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **55.83%** |
| Final Test Accuracy | **51.67%** |
| Macro F1-Score | **0.50** |

---

## 🎯 Project Objectives

The main objectives of this project are:

- Process raw speech audio using Python and Librosa.
- Convert speech signals into Mel-spectrogram representations.
- Build a Convolutional Neural Network for emotion classification.
- Improve feature learning using residual connections.
- Handle class imbalance using weighted Cross-Entropy Loss.
- Reduce overfitting using dropout and weight decay.
- Evaluate the model using multiple classification metrics.
- Test the model on completely unseen speakers.
- Analyze model performance using confusion matrices and F1-scores.

---

## 😊 Emotion Classes

The model classifies speech into the following 8 emotion categories:

| Label | Emotion | Samples |
|:---:|---|---:|
| 0 | Neutral | 96 |
| 1 | Calm | 192 |
| 2 | Happy | 192 |
| 3 | Sad | 192 |
| 4 | Angry | 192 |
| 5 | Fearful | 192 |
| 6 | Disgust | 192 |
| 7 | Surprised | 192 |

**Total Samples: 1,440**

---

## 🔀 Dataset Split

Instead of randomly splitting individual audio files, the dataset is divided based on **actor IDs**.

| Dataset | Actors | Samples |
|---|---|---:|
| Training | Actor 01–16 | 960 |
| Validation | Actor 17–20 | 240 |
| Testing | Actor 21–24 | 240 |

### Why Speaker-Based Splitting?

Speech contains speaker-specific characteristics such as:

- Voice pitch
- Speaking style
- Vocal characteristics
- Accent
- Tone

If recordings from the same speaker appear in both training and testing data, the model may learn speaker-specific characteristics instead of emotion-specific characteristics.

Therefore, a speaker-independent split provides a more realistic evaluation of model generalization.

---

## 🎵 Audio Preprocessing

The raw audio files are processed using **Librosa**.

### Preprocessing Configuration

| Parameter | Value |
|---|---:|
| Sampling Rate | 22,050 Hz |
| Audio Duration | 3 seconds |
| Number of Mel Bands | 64 |
| FFT Size | 2,048 |
| Hop Length | 512 |

### Preprocessing Steps

1. Load the audio file.
2. Resample the audio to 22,050 Hz.
3. Pad shorter audio files.
4. Truncate longer audio files to 3 seconds.
5. Generate the Mel-spectrogram.
6. Convert the Mel-spectrogram to decibel scale.
7. Normalize the features using training-set statistics.
8. Store the processed features as NumPy arrays.

### Mel-Spectrogram

Each audio sample is converted into:

```text
64 × 130
```

The Mel-spectrogram represents how the frequency content of speech changes over time.

---

## 🧠 Model Architecture

The final model is a **Custom Residual Convolutional Neural Network**.

### Architecture

```text
Input Mel-Spectrogram
        │
        ▼
Residual Block 1
1 → 32 Channels
        │
     MaxPool
        │
        ▼
Residual Block 2
32 → 64 Channels
        │
     MaxPool
        │
        ▼
Residual Block 3
64 → 128 Channels
        │
     MaxPool
        │
        ▼
Residual Block 4
128 → 256 Channels
        │
        ▼
Global Average Pooling
        │
        ▼
Fully Connected Layer
256 → 128
        │
      ReLU
        │
     Dropout
        │
        ▼
Output Layer
128 → 8
        │
        ▼
Emotion Prediction
```

### Residual Blocks

Each residual block contains:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Shortcut connection

The shortcut connection allows information to bypass convolutional layers and helps improve gradient flow during training.

---

## ⚙️ Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 32 |
| Maximum Epochs | 60 |
| Loss Function | Weighted Cross-Entropy Loss |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | 10 epochs |

### Class Weighting

The dataset contains fewer Neutral samples compared with the other emotion classes.

To reduce the effect of class imbalance, class weights are applied to the Cross-Entropy Loss.

---

## 🛡️ Overfitting Control

During training, the model reached very high training accuracy while validation accuracy remained significantly lower.

This indicates that the model was learning the training data much more effectively than unseen validation data.

The following techniques were used to control overfitting:

- Dropout
- Weight decay
- Learning-rate scheduling
- Early stopping
- Speaker-independent validation
- Best validation checkpoint selection

The best validation accuracy achieved was:

**55.83%**

---

## 📊 Final Evaluation Results

The final model was evaluated on **240 test samples from completely unseen speakers**.

### Overall Performance

| Metric | Score |
|---|---|
| Test Accuracy | **51.67%** |
| Macro Precision | **0.55** |
| Macro Recall | **0.53** |
| Macro F1-Score | **0.50** |
| Weighted Precision | **0.55** |
| Weighted Recall | **0.52** |
| Weighted F1-Score | **0.50** |

---

## 📋 Classification Report

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| Neutral | 0.43 | 0.75 | 0.55 |
| Calm | 0.72 | 0.56 | 0.63 |
| Happy | 0.31 | 0.56 | 0.40 |
| Sad | 0.29 | 0.16 | 0.20 |
| Angry | 0.58 | 0.66 | 0.62 |
| Fearful | 0.71 | 0.16 | 0.26 |
| Disgust | 0.71 | 0.69 | 0.70 |
| Surprised | 0.62 | 0.72 | 0.67 |

---

## 📈 Training Performance

The training process records both training and validation accuracy/loss.

The resulting visualization is stored at:

```text
results/residual_training_curves.png
```

The training curves help analyze:

- Model convergence
- Training performance
- Validation performance
- Overfitting
- Learning-rate changes

The model achieved very high training accuracy while validation accuracy remained lower, showing that the model was prone to overfitting.

Early stopping and best-validation-checkpoint selection were used to address this.

---

## 🔥 Confusion Matrix

The confusion matrix is stored at:

```text
results/confusion_matrix.png
```

The confusion matrix provides a class-by-class view of model predictions.

It helps identify:

- Correct predictions
- Incorrect predictions
- Frequently confused emotions
- Difficult emotion classes

The model has more difficulty distinguishing emotions such as:

- Happy
- Sad
- Fearful

while **Disgust, Surprised, Angry, and Calm** show stronger performance.

---

## 📊 F1-Score Analysis

The F1-score visualization is stored at:

```text
results/f1_scores.png
```

F1-score combines precision and recall and is useful for evaluating the performance of individual emotion classes.

### Highest F1-Scores

| Emotion | F1-Score |
|---|---|
| Disgust | **0.70** |
| Surprised | **0.67** |
| Calm | **0.63** |
| Angry | **0.62** |

### Most Difficult Classes

| Emotion | F1-Score |
|---|---|
| Sad | **0.20** |
| Fearful | **0.26** |
| Happy | **0.40** |

---

## 🧪 Experiments

Several approaches were tested during development before selecting the final model.

### 1. Standard CNN

A basic CNN using Mel-spectrogram features was implemented as the initial baseline.

The baseline provided a useful reference for comparing more advanced architectures.

### 2. Data Augmentation

Audio/spectrogram augmentation techniques were tested.

The tested augmentation configuration reduced model performance, so it was not selected for the final pipeline.

### 3. 5-Second Audio

A 5-second audio configuration was tested.

The 5-second configuration performed worse than the 3-second configuration, so the 3-second representation was retained.

### 4. MFCC + Delta + Delta-Delta

MFCC features together with delta and delta-delta features were tested.

This approach did not improve validation performance compared with the Mel-spectrogram approach.

### 5. ResNet18 Transfer Learning

A pretrained ResNet18-based approach was also tested.

The transfer-learning approach did not outperform the custom Residual CNN.

### Final Model

After comparing the different approaches, the custom Residual CNN using 3-second Mel-spectrograms was selected as the final model.

```text
RAVDESS Audio
      ↓
3-Second Audio
      ↓
Mel-Spectrogram
      ↓
Normalization
      ↓
Residual CNN
      ↓
8 Emotion Classes
```

---

## 🛠️ Technical Approach

### Python

Used as the primary programming language for the complete machine learning pipeline.

### Librosa

Used for:

- Audio loading
- Resampling
- Mel-spectrogram extraction
- Audio preprocessing

### PyTorch

Used for:

- CNN implementation
- Residual blocks
- Model training
- Optimization
- Loss calculation
- Model evaluation

### NumPy

Used for:

- Numerical operations
- Feature arrays
- Dataset storage

### Scikit-learn

Used for:

- Classification reports
- Precision
- Recall
- F1-score
- Confusion matrix

### Matplotlib

Used for:

- Training curves
- Confusion matrix visualization
- F1-score visualization

---

## 📂 Repository Structure

```text
Speech Recognition/
│
├── README.md
│
├── dataset/
│   └── processed/
│       ├── X.npy
│       ├── y.npy
│       ├── actors.npy
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_val.npy
│       ├── y_val.npy
│       ├── X_test.npy
│       └── y_test.npy
│
├── models/
│   ├── emotion_cnn_residual_best.pth
│   └── residual_training_history.json
│
├── results/
│   ├── residual_training_curves.png
│   ├── confusion_matrix.png
│   └── f1_scores.png
│
└── src/
    ├── check_dataset.py
    ├── check_durations.py
    ├── check_labels.py
    ├── create_dataset.py
    ├── evaluate.py
    ├── plot_results.py
    ├── split_dataset.py
    └── train.py
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd "Speech Recognition"
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

For Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install numpy librosa torch torchvision scikit-learn matplotlib
```

### 5. Add the RAVDESS Dataset

The RAVDESS audio dataset is not included in this repository.

After downloading the dataset, place the actor folders inside:

```text
dataset/RAVDESS/
```

Expected structure:

```text
dataset/RAVDESS/
├── Actor_01/
├── Actor_02/
├── Actor_03/
├── ...
└── Actor_24/
```

### 6. Check the Dataset

```bash
python src/check_dataset.py
```

### 7. Check Audio Durations

```bash
python src/check_durations.py
```

### 8. Check Emotion Labels

```bash
python src/check_labels.py
```

### 9. Create the Processed Dataset

```bash
python src/create_dataset.py
```

### 10. Create the Speaker-Based Split

```bash
python src/split_dataset.py
```

### 11. Train the Model

```bash
python src/train.py
```

### 12. Evaluate the Model

```bash
python src/evaluate.py
```

### 13. Generate Detailed Evaluation

```bash
python src/evaluate_detailed.py
```

### 14. Generate Result Visualizations

```bash
python src/plot_results.py
```

---

## 🔄 End-to-End Workflow

```text
                    RAVDESS Dataset
                           │
                           ▼
                  Audio Preprocessing
                           │
                           ▼
                    3-Second Audio
                           │
                           ▼
                    Mel-Spectrogram
                           │
                           ▼
                  Feature Normalization
                           │
                           ▼
                Speaker-Based Data Split
                    /       |       \
                   /        |        \
                  ▼         ▼         ▼
             Training   Validation   Testing
                │           │           │
                ▼           │           │
           Residual CNN     │           │
                │           │           │
                ▼           │           │
             Training       │           │
                │           │           │
                ▼           │           │
        Best Validation     │           │
           Checkpoint       │           │
                │           │           │
                └───────────┴───────────┘
                            │
                            ▼
                    Unseen-Speaker Test
                            │
                            ▼
                       Evaluation
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          Accuracy       F1-Score   Confusion Matrix
```

---

## 📌 Important Notes

- The original RAVDESS audio dataset is **not included** in this repository.
- The dataset must be downloaded separately.
- The audio files should be placed inside `dataset/RAVDESS/`.
- The final model uses a speaker-independent evaluation setup.
- The test speakers are not present in the training set.
- The final reported test accuracy is **51.67%**.

### Speaker Split

```text
Training   → Actors 01–16
Validation → Actors 17–20
Testing    → Actors 21–24
```

This ensures that the final test performance measures generalization to unseen speakers.

---

## 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

- Speech Emotion Recognition
- Audio signal processing
- Mel-spectrogram feature extraction
- Deep learning with PyTorch
- Convolutional Neural Networks
- Residual Neural Networks
- Residual connections
- Batch normalization
- Dropout
- L2-style regularization
- AdamW optimization
- Class-weighted loss
- Learning-rate scheduling
- Early stopping
- Speaker-independent evaluation
- Confusion matrix analysis
- Precision, recall, and F1-score analysis
- Model comparison
- Deep learning experimentation
- Overfitting analysis

---

## 📈 Possible Next Steps

Future improvements could include:

- Using larger and more diverse speech emotion datasets.
- Experimenting with pretrained speech models.
- Exploring transformer-based speech representations.
- Adding attention mechanisms.
- Improving performance on difficult emotions such as Sad and Fearful.
- Performing systematic hyperparameter optimization.
- Testing carefully selected audio augmentation techniques.
- Investigating speaker-invariant feature learning.
- Deploying the trained model as a real-time emotion recognition application.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| PyTorch | Deep Learning |
| Librosa | Audio Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Model Evaluation |
| Matplotlib | Data Visualization |
| VS Code | Development |

---

## ⭐ Project Summary

| Component | Details |
|---|---|
| Project | Speech Emotion Recognition |
| Dataset | RAVDESS |
| Audio Files | 1,440 |
| Actors | 24 |
| Emotion Classes | 8 |
| Audio Duration | 3 seconds |
| Sampling Rate | 22,050 Hz |
| Feature | Mel-Spectrogram |
| Feature Size | 64 × 130 |
| Model | Custom Residual CNN |
| Framework | PyTorch |
| Training Samples | 960 |
| Validation Samples | 240 |
| Test Samples | 240 |
| Best Validation Accuracy | **55.83%** |
| Final Test Accuracy | **51.67%** |
| Macro F1-Score | **0.50** |

---

## 👨‍💻 Author

**Praveena Kamanuru**
