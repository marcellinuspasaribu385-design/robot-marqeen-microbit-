import time
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("results/best_model_partial.pth")
TEST_IMAGE = next(Path("data/split/val").glob("*/*.jpg"), None)

if TEST_IMAGE is None:
    TEST_IMAGE = next(Path("data/split/val").glob("*/*.png"), None)

if not MODEL_PATH.exists():
    raise FileNotFoundError("Jalankan train.py terlebih dahulu.")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

if TEST_IMAGE:
    img = Image.open(TEST_IMAGE).convert("RGB")
else:
    img = Image.new("RGB", (224,224))

x = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    for _ in range(10):
        _ = model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        _ = model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

ms = elapsed / n * 1000
fps = 1000 / ms
print(f"Device: {DEVICE}")
print(f"Average inference latency: {ms:.2f} ms/frame")
print(f"Estimated inference FPS: {fps:.2f}")
