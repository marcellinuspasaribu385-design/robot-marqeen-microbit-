import argparse
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["feature", "partial", "scratch", "all"], default="all")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = Path("data/split")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.ImageFolder(DATA / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA / "val", transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

NUM_CLASSES = len(train_ds.classes)
print("Classes:", train_ds.classes)
print("Device:", DEVICE)

def make_model(mode):
    if mode == "scratch":
        m = models.resnet18(weights=None)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if mode == "feature":
        for p in m.parameters():
            p.requires_grad = False
    elif mode == "partial":
        for p in m.parameters():
            p.requires_grad = False
        for p in m.layer4.parameters():
            p.requires_grad = True

    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)

def set_frozen_bn_eval(model, mode):
    if mode == "scratch":
        return
    # Frozen backbone tetap eval agar statistik BatchNorm pretrained tidak berubah.
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if mode == "feature" or (mode == "partial" and not name.startswith("layer4")):
                module.eval()

def optimizer_for(model, mode):
    if mode == "partial":
        return Adam([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ])
    return Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

def run(mode):
    model = make_model(mode)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_for(model, mode)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        set_frozen_bn_eval(model, mode)
        correct = total = 0
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total if total else 0

        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total else 0
        scheduler.step()

        history.append({
            "mode": mode, "epoch": epoch + 1,
            "train_loss": train_loss / max(1, len(train_ds)),
            "train_acc": train_acc,
            "val_loss": val_loss / max(1, len(val_ds)),
            "val_acc": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"[{mode}] epoch {epoch+1}/{args.epochs} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    elapsed = time.time() - start
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(),
        "classes": train_ds.classes,
        "mode": mode,
        "best_val_acc": best_acc,
    }, RESULTS / f"best_model_{mode}.pth")

    return history, best_acc, elapsed

modes = ["feature", "partial", "scratch"] if args.mode == "all" else [args.mode]
all_hist, summary = [], []

for mode in modes:
    hist, best_acc, elapsed = run(mode)
    all_hist.extend(hist)
    summary.append({
        "mode": mode,
        "best_val_accuracy": best_acc,
        "training_time_seconds": elapsed,
    })

pd.DataFrame(all_hist).to_csv(RESULTS / "metrics.csv", index=False)
pd.DataFrame(summary).to_csv(RESULTS / "summary.csv", index=False)

df = pd.DataFrame(all_hist)
plt.figure()
for mode in modes:
    d = df[df["mode"] == mode]
    plt.plot(d.epoch, d.val_acc, marker="o", label=mode)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("ResNet-18 Transfer Learning - Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS / "accuracy_curve.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nRingkasan:")
print(pd.DataFrame(summary))
