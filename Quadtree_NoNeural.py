import torch
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# =====================================================
# DATA
# =====================================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# =====================================================
# QUADTREE FEATURES
# =====================================================

def quadtree_features(img, max_depth=3):

    features = []

    def recurse(block, depth):

        mean = block.mean().item()
        std = block.std().item()

        features.append(mean)
        features.append(std)

        if depth >= max_depth:
            return

        h = block.shape[0]
        w = block.shape[1]

        if h < 2 or w < 2:
            return

        h2 = h // 2
        w2 = w // 2

        recurse(block[:h2, :w2], depth + 1)
        recurse(block[:h2, w2:], depth + 1)
        recurse(block[h2:, :w2], depth + 1)
        recurse(block[h2:, w2:], depth + 1)

    recurse(img, 0)

    return np.array(features)

# =====================================================
# BUILD FEATURE MATRICES
# =====================================================

print("Extracting training features...")

X_train = []
y_train = []

for img, label in train_dataset:

    img = img.squeeze()

    X_train.append(
        quadtree_features(img)
    )

    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Train shape:", X_train.shape)

print("Extracting test features...")

X_test = []
y_test = []

for img, label in test_dataset:

    img = img.squeeze()

    X_test.append(
        quadtree_features(img)
    )

    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Test shape:", X_test.shape)

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

print("Training Logistic Regression...")

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

clf.fit(
    X_train,
    y_train
)

# =====================================================
# EVALUATION
# =====================================================

preds = clf.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print("\nTest Accuracy:", acc * 100)

# =====================================================
# FEATURE INFORMATION
# =====================================================

print("\nOriginal variables :", 28 * 28)
print("Compressed variables:", X_train.shape[1])

compression_ratio = (
    X_train.shape[1]
    / (28 * 28)
)

print(
    "Compression ratio:",
    compression_ratio
)
