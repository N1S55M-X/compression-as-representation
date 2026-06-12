import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor()
])

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
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# POLAR COMPRESSION LAYER
# ============================================================

class PolarCompression(nn.Module):

    def __init__(self, out_features=128):
        super().__init__()

        self.out_features = out_features

        # Learnable projection matrix
        self.projection = nn.Linear(
            3,
            out_features
        )

    def forward(self, x):

        # x shape = [B,1,28,28]
        B, C, H, W = x.shape

        device = x.device

        # Coordinate grid
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )

        # Convert coordinates to polar
        r = torch.sqrt(xs**2 + ys**2)

        theta = torch.atan2(
            ys,
            xs
        )

        # Polar features
        polar_features = torch.stack(
            [
                r,
                torch.cos(theta),
                torch.sin(theta)
            ],
            dim=-1
        )

        # [784,3]
        polar_features = polar_features.view(-1, 3)

        # Learn basis vectors
        # [784,128]
        basis = self.projection(
            polar_features
        )

        # Flatten pixels
        # [B,784]
        pixels = x.view(B, -1)

        # Compression
        # [B,128]
        compressed = torch.matmul(
            pixels,
            basis
        )

        return compressed


# ============================================================
# MODEL
# ============================================================

class PolarNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.compress = PolarCompression(
            out_features=128
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                10
            )
        )

    def forward(self, x):

        x = self.compress(x)

        x = self.classifier(x)

        return x


# ============================================================
# MODEL SETUP
# ============================================================

model = PolarNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
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

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_acc = (
        100 * correct / total
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss={running_loss/len(train_loader):.4f} | "
        f"Train Acc={train_acc:.2f}%"
    )

# ============================================================
# TEST
# ============================================================

print("\nEvaluating...\n")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

test_acc = (
    100 * correct / total
)

print(
    f"\nFinal Test Accuracy: "
    f"{test_acc:.2f}%"
)

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "polar_mnist_model.pth"
)

print(
    "\nModel saved as polar_mnist_model.pth"
)

