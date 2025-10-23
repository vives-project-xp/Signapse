import os
import json
import random
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Dataset wortel (na jouw download & extract scripts)
DATASET_ROOT = "./notebooks/images/dataset"   # mappen: 0..9, a..z
ASSERT_JSON = ("landmarks.json", "landmarks_all.json")
label_map = sorted(d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)))
print("Labels:", label_map)

classes = label_map
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
print("Class to index mapping:", class_to_idx)

os.makedirs("./annotations", exist_ok=True)
with open("./annotations/label_map.json", "w") as f:
    json.dump({"class_to_idx": class_to_idx}, f, indent=2)
# Cell 2: preprocess helpers

MCP_IDX = [5, 9, 13, 17]   # niet strikt nodig, maar handig
def preprocess_landmarks(lms: List[Dict[str, float]]) -> np.ndarray:
    pts = np.array([[d["x"], d["y"], d.get("z", 0.0)] for d in lms], dtype=np.float32)  # (21,3)
    wrist_xy = pts[0, :2].copy()
    pts[:, :2] -= wrist_xy
    span = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    pts[:, :2] /= span
    pts[:, 2]  /= span
    return pts.flatten()  # (63,)

def maybe_load_json(path_dir: str) -> Optional[List[Dict[str, Any]]]:
    for name in ASSERT_JSON:
        p = os.path.join(path_dir, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

# Cell 3: laad alle JSONs en vorm X, y
X, y, metas = [], [], []  # metas: (label, image_file)
missing = []

for label in classes:
    folder = os.path.join(DATASET_ROOT, label)
    data = maybe_load_json(folder)
    if not data:
        missing.append(label)
        continue
    for entry in data:
        lms = entry.get("landmarks")
        if not lms or len(lms) != 21:
            continue
        feat = preprocess_landmarks(lms)  # 63D
        X.append(feat)
        y.append(class_to_idx[label])
        metas.append((label, entry.get("image") or entry.get("image_file") or ""))
        
X = np.stack(X).astype(np.float32)
y = np.array(y, dtype=np.int64)
print("Samples:", len(y), "Dim:", X.shape, "Missings(json):", missing[:3], "…")
assert X.shape[1] == 63

# Cell 4: stratified splits
test_size = 0.15
val_size  = 0.15

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
train_val_idx, test_idx = next(sss1.split(X, y))

X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Val t.o.v. totale set => verhouding op train_val bepalen:
val_ratio_rel = val_size / (1.0 - test_size)  # bv. 0.15 / 0.85
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_rel, random_state=SEED)
train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))

X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
X_val,   y_val   = X_train_val[val_idx], y_train_val[val_idx]

for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(name, "size:", len(yy))


# Cell 5: torch datasets/dataloaders
class LandmarkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.augment:
            # lichte jitter op x,y,z (mean 0, std 0.01)
            noise = np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            x += noise
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

BATCH = 256
train_ds = LandmarkDataset(X_train, y_train, augment=True)
val_ds   = LandmarkDataset(X_val,   y_val,   augment=False)
test_ds  = LandmarkDataset(X_test,  y_test,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, drop_last=False)

# Cell 6: model + training
N_FEAT = 63
N_CLS  = len(classes)

class MLP(nn.Module):
    def __init__(self, in_dim=N_FEAT, n_classes=N_CLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):  # x: (B,63)
        return self.net(x)

model = MLP().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
crit = nn.CrossEntropyLoss()

def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, preds, gts = 0.0, [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True).float()
        yb = yb.to(DEVICE, non_blocking=True)
        if train:
            opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        if train:
            loss.backward()
            opt.step()
        total_loss += loss.item() * yb.size(0)
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        gts.extend(yb.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(gts, preds)
    f1m = f1_score(gts, preds, average="macro", zero_division=0)
    return avg_loss, acc, f1m

BEST_VAL = math.inf
PATIENCE, bad = 10, 0
EPOCHS = 50

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, True)
    va_loss, va_acc, va_f1 = run_epoch(val_loader, False)
    sched.step(va_loss)
    print(f"ep{ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} "
          f"| val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

    if va_loss < BEST_VAL - 1e-4:
        BEST_VAL = va_loss
        bad = 0
        torch.save(model.state_dict(), "./annotations/asl_mlp_best.pth")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("Early stopping.")
            break

# laad best
model.load_state_dict(torch.load("./annotations/asl_mlp_best.pth", map_location=DEVICE))
model.eval()

# Cell 7: evaluatie
def predict_all(loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE).float()
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = predict_all(test_loader)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))
labels = sorted(np.unique(y_true))
print(classification_report(
    y_true, y_pred,
    labels=labels,
    target_names=[idx_to_class[i] for i in labels],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLS)))
fig = plt.figure(figsize=(10, 9))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (test)")
plt.colorbar()
tick_marks = np.arange(N_CLS)
plt.xticks(tick_marks, [idx_to_class[i] for i in range(N_CLS)], rotation=90)
plt.yticks(tick_marks, [idx_to_class[i] for i in range(N_CLS)])
plt.tight_layout()
plt.xlabel('Predicted'); plt.ylabel('True')
plt.show()

# Cell 8: export
# state_dict
torch.save(model.state_dict(), "./annotations/asl_mlp_state_dict.pth")

# TorchScript (script)
scripted = torch.jit.script(model.cpu())
scripted.save("./annotations/asl_mlp_script.pt")

# ONNX
dummy = torch.randn(1, 63, dtype=torch.float32)
torch.onnx.export(
    model.cpu(), dummy, "./annotations/asl_mlp.onnx",
    input_names=["x"], output_names=["logits"],
    dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17
)
print("Exported to ./annotations/")

# Cell 9: webcam inference
# Zorg dat mediapipe en opencv beschikbaar zijn.
mp_hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, model_complexity=1,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

def preprocess_from_mp(lmk_list) -> np.ndarray:
    # zelfde normalisatie als tijdens training
    pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lmk_list], dtype=np.float32)
    wrist_xy = pts[0, :2].copy()
    pts[:, :2] -= wrist_xy
    span = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    pts[:, :2] /= span
    pts[:, 2]  /= span
    return pts.flatten()

# Herlaad model en labelmap op CPU of GPU
model.load_state_dict(torch.load("./annotations/asl_mlp_best.pth", map_location=DEVICE))
model.eval().to(DEVICE)
with open("./annotations/label_map.json", "r") as f:
    class_to_idx = json.load(f)["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("[ERR] camera niet open")
else:
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        pred_label = None

        if res.multi_hand_landmarks:
            lmk = res.multi_hand_landmarks[0].landmark
            feat = preprocess_from_mp(lmk)
            with torch.no_grad():
                xb = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
                logits = model(xb.float())
                prob = torch.softmax(logits, dim=1)
                idx = int(torch.argmax(prob, dim=1).item())
                pred_label = idx_to_class[idx]
                conf = float(prob[0, idx].cpu())
            # teken landmarks
            mp_drawing = cv
            for lm in lmk:
                h, w = frame.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(frame, (cx, cy), 3, (0,255,0), -1)
            cv.putText(frame, f"{pred_label} ({conf:.2f})", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow("ASL landmarks classifier (MLP)", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
