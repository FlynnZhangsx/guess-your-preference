"""
Training v7 — Per-User Stratified Split + BCE Loss
----------------------------------------------------
Key improvements from reference:
- Per-user image stratification (train/val/test on images, not users)
- BCEWithLogitsLoss with pos_weight for class balance
- Multi-metric evaluation: AUC, AP, Acc, Hit@K, MRR, Pairwise
- ReduceLROnPlateau scheduler + Early Stopping
- Residual MLP (768→384→768) with ViT-L/14 features
"""

import os
import json
import math
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from model import PreferenceAlignModel
from dataset import PreferenceDataset

# ============================================================
# Config
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_v7")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "preference_model_v7.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cpu"
DIM = 768
HIDDEN_DIM = 384
DROPOUT = 0.3

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100

TEMPERATURE = 0.07
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NEG_RATIO = 3
PATIENCE = 8
MIN_LR = 1e-6
SEED = 42

# ============================================================
# Setup
# ============================================================

print("=" * 60)
print("Preference Alignment v7 — Stratified Split + BCE Loss")
print("=" * 60)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Datasets
print("\n[1] Building stratified datasets...")
train_ds = PreferenceDataset(DATA_CACHE_DIR, split='train', seed=SEED,
                              val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                              neg_ratio=NEG_RATIO)
val_ds = PreferenceDataset(DATA_CACHE_DIR, split='val', seed=SEED,
                            val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                            neg_ratio=NEG_RATIO)
test_ds = PreferenceDataset(DATA_CACHE_DIR, split='test', seed=SEED,
                             val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                             neg_ratio=NEG_RATIO)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# pos_weight for BCE
n_pos = sum(1 for s in train_ds.samples if s[3] == 1.0)
n_neg = sum(1 for s in train_ds.samples if s[3] == 0.0)
pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))])
print(f"  Pos/Neg: {n_pos}/{n_neg}, pos_weight={pos_weight.item():.3f}")

# Model
print("\n[2] Initializing model...")
model = PreferenceAlignModel(dim=DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {total_params:,} | Dim: {DIM}→{HIDDEN_DIM}→{DIM} | Temp: {TEMPERATURE}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, min_lr=MIN_LR
)

# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(model, loader):
    """Compute comprehensive evaluation metrics."""
    model.eval()
    all_labels = []
    all_scores = []
    user_scores = defaultdict(list)  # user_idx -> [(label, score)]

    for batch in loader:
        text_f = batch[0].to(DEVICE)
        img_f = batch[1].to(DEVICE)
        labels = batch[2]

        text_proj = model.encode_text(text_f)
        img_proj = model.encode_image(img_f)
        sim = (text_proj * img_proj).sum(dim=-1) / TEMPERATURE

        all_labels.append(labels.numpy())
        all_scores.append(sim.cpu().numpy())

        # Per-user tracking: need user_idx
        # We need to track which user each sample belongs to
        # This requires knowing batch indices...

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)
    y_prob = 1.0 / (1.0 + np.exp(-y_score))  # sigmoid

    metrics = {}
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    except:
        metrics['auc'] = float('nan')
    try:
        metrics['ap'] = float(average_precision_score(y_true, y_prob))
    except:
        metrics['ap'] = float('nan')
    y_pred = (y_score >= 0.0).astype(np.float32)
    try:
        metrics['acc'] = float(accuracy_score(y_true, y_pred))
    except:
        metrics['acc'] = float('nan')

    return metrics


@torch.no_grad()
def compute_full_metrics(model, dataset):
    """
    Per-user ranking metrics: for each user, rank all their val/test
    images and compute Hit@K, MRR, Pairwise Accuracy.
    """
    model.eval()
    user_rankings = defaultdict(list)  # user_idx -> [(label, score)]

    # Process all samples
    all_text = []
    all_img = []
    all_labels = []
    all_users = []

    for idx in range(len(dataset)):
        user_idx, img_key, _, label = dataset.samples[idx]
        t = dataset.text_features[user_idx][0]  # variant 0
        v = dataset.image_features[img_key][0]  # variant 0
        all_text.append(t)
        all_img.append(v)
        all_labels.append(label)
        all_users.append(user_idx)

    # Batch compute
    text_batch = torch.stack(all_text).to(DEVICE)
    img_batch = torch.stack(all_img).to(DEVICE)

    # Process in sub-batches to avoid OOM
    scores = []
    for i in range(0, len(text_batch), 512):
        tb = text_batch[i:i+512]
        ib = img_batch[i:i+512]
        tp = model.encode_text(tb)
        ip = model.encode_image(ib)
        scores.append(((tp * ip).sum(dim=-1) / TEMPERATURE).cpu())

    all_scores = torch.cat(scores).numpy()

    for u, l, s in zip(all_users, all_labels, all_scores):
        user_rankings[u].append((int(l), float(s)))

    # Compute per-user metrics
    ks = [1, 3, 5]
    hits = {k: [] for k in ks}
    mrrs = []
    pairwise_accs = []

    for uid, items in user_rankings.items():
        labels = np.array([x[0] for x in items])
        scores = np.array([x[1] for x in items])

        order = np.argsort(-scores)
        sorted_labels = labels[order]
        pos_positions = np.where(sorted_labels == 1)[0]

        for k in ks:
            hits[k].append(float(np.any(pos_positions < k)))
        mrrs.append(1.0 / float(pos_positions[0] + 1) if len(pos_positions) > 0 else 0.0)

        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if len(pos_scores) and len(neg_scores):
            comparisons = (pos_scores[:, None] > neg_scores[None, :]).astype(np.float32)
            pairwise_accs.append(float(comparisons.mean()))

    out = {f'hit@{k}': float(np.mean(hits[k])) if hits[k] else float('nan') for k in ks}
    out['mrr'] = float(np.mean(mrrs)) if mrrs else float('nan')
    out['pairwise_acc'] = float(np.mean(pairwise_accs)) if pairwise_accs else float('nan')
    return out


# ============================================================
# Training
# ============================================================

print("\n[3] Training...")
print("=" * 60)

best_val_auc = -1.0
best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
bad_epochs = 0
history = {'train_loss': [], 'val_auc': [], 'val_acc': [], 'val_pairwise': []}

pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training")
for epoch in pbar:
    # --- Train ---
    model.train()
    epoch_loss = 0.0
    n_samples = 0

    for batch in train_loader:
        text_f = batch[0].to(DEVICE)
        img_f = batch[1].to(DEVICE)
        labels = batch[2].to(DEVICE)

        text_proj = model.encode_text(text_f)
        img_proj = model.encode_image(img_f)
        logits = (text_proj * img_proj).sum(dim=-1) / TEMPERATURE

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(labels)
        n_samples += len(labels)

    avg_loss = epoch_loss / max(1, n_samples)
    history['train_loss'].append(avg_loss)

    # --- Evaluate ---
    val_metrics = compute_metrics(model, val_loader)
    val_auc = val_metrics.get('auc', float('nan'))

    history['val_auc'].append(val_auc)
    history['val_acc'].append(val_metrics.get('acc', float('nan')))

    if not math.isnan(val_auc):
        scheduler.step(val_auc)

    # Full ranking metrics every 5 epochs
    if epoch % 5 == 0 or epoch == 1 or epoch == NUM_EPOCHS:
        rank_metrics = compute_full_metrics(model, val_ds)
        history['val_pairwise'].append(rank_metrics.get('pairwise_acc', float('nan')))
    else:
        rank_metrics = {'pairwise_acc': float('nan'), 'hit@1': float('nan'), 'mrr': float('nan')}

    pbar.set_postfix({
        'Loss': f'{avg_loss:.4f}',
        'AUC': f'{val_auc:.4f}',
        'PW': f'{rank_metrics.get("pairwise_acc", 0):.3f}',
        'Best': f'{best_val_auc:.4f}',
    })

    if epoch % 10 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(
            f"  Epoch {epoch:3d} | LR: {current_lr:.2e} | Loss: {avg_loss:.4f} | "
            f"AUC: {val_auc:.4f} | Acc: {val_metrics.get('acc', 0):.4f} | "
            f"PW: {rank_metrics.get('pairwise_acc', 0):.4f} | "
            f"H@1: {rank_metrics.get('hit@1', 0):.4f} | MRR: {rank_metrics.get('mrr', 0):.4f}"
        )

    # Save best
    if not math.isnan(val_auc) and val_auc > best_val_auc:
        best_val_auc = val_auc
        bad_epochs = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
            'history': history,
        }, best_path)
        tqdm.write(f"  -> Best model saved (AUC={val_auc:.4f})")
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        tqdm.write(f"  Early stopping at epoch {epoch}")
        break

# ============================================================
# Final Evaluation
# ============================================================

print("\n" + "=" * 60)
print("[4] Final evaluation...")
print("=" * 60)

# Load best
if os.path.exists(best_path):
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded best model (epoch {ckpt['epoch']}, AUC={ckpt['val_auc']:.4f})")

# Test metrics
print("\n  --- Test Set ---")
test_metrics = compute_metrics(model, test_loader)
print(f"  AUC: {test_metrics.get('auc', float('nan')):.4f}")
print(f"  AP:  {test_metrics.get('ap', float('nan')):.4f}")
print(f"  Acc: {test_metrics.get('acc', float('nan')):.4f}")

test_rank = compute_full_metrics(model, test_ds)
print(f"  Hit@1: {test_rank.get('hit@1', float('nan')):.4f}")
print(f"  Hit@3: {test_rank.get('hit@3', float('nan')):.4f}")
print(f"  Hit@5: {test_rank.get('hit@5', float('nan')):.4f}")
print(f"  MRR:   {test_rank.get('mrr', float('nan')):.4f}")
print(f"  Pairwise Acc: {test_rank.get('pairwise_acc', float('nan')):.4f}")

# Save final
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'test_metrics': test_metrics,
    'test_rank': test_rank,
}, MODEL_SAVE_PATH)

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), 'w') as f:
    json.dump({**test_metrics, **test_rank}, f, indent=2)

print(f"\n  Model: {MODEL_SAVE_PATH}")
print(f"  Best Val AUC: {best_val_auc:.4f}")
print("=" * 60)
print("V7 training complete!")
print("=" * 60)
