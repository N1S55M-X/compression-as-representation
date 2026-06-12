import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", DEVICE)

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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

# =====================================================
# QUADTREE FEATURE EXTRACTOR
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

    return torch.tensor(features)


# =====================================================
# DETERMINE FEATURE SIZE
# =====================================================

sample_img = train_dataset[0][0].squeeze()

FEATURE_SIZE = len(
    quadtree_features(sample_img)
)

print("Feature size =", FEATURE_SIZE)

# =====================================================
# MODEL
# =====================================================

class QuadtreeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                FEATURE_SIZE,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        batch_features = []

        for img in x:

            img = img.squeeze()

            feat = quadtree_features(img)

            batch_features.append(feat)

        batch_features = torch.stack(
            batch_features
        ).to(x.device)

        return self.net(
            batch_features
        )


# =====================================================
# TRAIN SETUP
# =====================================================

model = QuadtreeNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# =====================================================
# TRAIN
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    print(
        f"Epoch {epoch+1} | "
        f"Loss={total_loss/len(train_loader):.4f} | "
        f"Acc={100*correct/total:.2f}%"
    )

# =====================================================
# TEST
# =====================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

print(
    "\nTest Accuracy:",
    100 * correct / total
)

torch.save(
    model.state_dict(),
    "quadtree_mnist.pth"
)

print(
    "Saved: quadtree_mnist.pth"
)
