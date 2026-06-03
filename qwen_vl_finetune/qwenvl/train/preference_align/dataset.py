"""
Preference Dataset v7 — Per-User Stratified Split
----------------------------------------------------
Each user's positive and negative images are independently split into
train/val/test. The model sees all users during training but only a
subset of their images, testing generalization to unseen images.

Augmented features: text [3, 768] variants, image [K, 768] variants.
Train: random variant sampling + noise + cutout
Eval: variant 0 (original), no noise
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    """Flat dataset of (text_feat, img_feat, label) pairs for BCE loss."""

    def __init__(self, data_cache_dir, split='train', seed=42,
                 cutout_ratio=0.10, noise_std=0.01,
                 val_ratio=0.15, test_ratio=0.15, neg_ratio=3):
        super().__init__()
        self.data_cache_dir = data_cache_dir
        self.split = split
        self.train = (split == 'train')
        self.cutout_ratio = cutout_ratio if self.train else 0.0
        self.noise_std = noise_std if self.train else 0.0
        self.rng = random.Random(seed)

        # Load features
        text_path = os.path.join(data_cache_dir, "aug_text_features.pt")
        image_path = os.path.join(data_cache_dir, "aug_image_features.pt")
        mapping_path = os.path.join(data_cache_dir, "dataset_mapping.json")

        for p in [text_path, image_path, mapping_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found. Run extract_features.py first.")

        print(f"[Dataset v7] Loading features ({split})...")
        self.text_features = torch.load(text_path, map_location='cpu', weights_only=True)
        self.image_features = torch.load(image_path, map_location='cpu', weights_only=True)

        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)

        # Build stratified per-user splits
        self.samples = []  # list of (user_idx, pos_img_key, neg_img_key)
        rng_split = random.Random(seed + 1)

        for user_idx, entry in enumerate(self.mapping):
            pos_keys = entry['pos_image_keys']
            neg_keys = entry['neg_image_keys']

            if len(pos_keys) < 2 or len(neg_keys) < 1:
                continue

            # Shuffle and split positives
            pos_shuffled = pos_keys.copy()
            rng_split.shuffle(pos_shuffled)
            n_pos = len(pos_shuffled)
            n_pos_test = max(1, int(round(n_pos * test_ratio)))
            n_pos_val = max(1, int(round(n_pos * val_ratio)))
            n_pos_train = n_pos - n_pos_test - n_pos_val
            if n_pos_train < 1:
                n_pos_train = 1
                n_pos_val = max(0, n_pos - n_pos_train - n_pos_test)

            pos_train = pos_shuffled[:n_pos_train]
            pos_val = pos_shuffled[n_pos_train:n_pos_train + n_pos_val]
            pos_test = pos_shuffled[n_pos_train + n_pos_val:]

            # Shuffle and split negatives
            neg_shuffled = neg_keys.copy()
            rng_split.shuffle(neg_shuffled)
            n_neg = len(neg_shuffled)
            n_neg_test = max(1, int(round(n_neg * test_ratio)))
            n_neg_val = max(1, int(round(n_neg * val_ratio)))
            n_neg_train = n_neg - n_neg_test - n_neg_val
            if n_neg_train < 1:
                n_neg_train = 1
                n_neg_val = max(0, n_neg - n_neg_train - n_neg_test)

            neg_train = neg_shuffled[:n_neg_train]
            neg_val = neg_shuffled[n_neg_train:n_neg_train + n_neg_val]
            neg_test = neg_shuffled[n_neg_train + n_neg_val:]

            # Store per-user split info
            if not hasattr(self, '_user_splits'):
                self._user_splits = {}
            self._user_splits[user_idx] = {
                'pos_train': pos_train, 'neg_train': neg_train,
                'pos_val': pos_val, 'neg_val': neg_val,
                'pos_test': pos_test, 'neg_test': neg_test,
            }

            # Select split
            if split == 'train':
                pos_pool = pos_train
                neg_pool = neg_train
            elif split == 'val':
                pos_pool = pos_val
                neg_pool = neg_val
            else:  # test
                pos_pool = pos_test
                neg_pool = neg_test

            if not pos_pool or not neg_pool:
                continue

            # Create positive pairs
            for pk in pos_pool:
                self.samples.append((user_idx, pk, None, 1.0))  # label=1 (positive)

            # Create negative pairs (subsample to neg_ratio * n_pos)
            max_negs = max(len(pos_pool) * neg_ratio, len(pos_pool))
            neg_sampled = rng_split.sample(neg_pool, min(len(neg_pool), max_negs))
            for nk in neg_sampled:
                self.samples.append((user_idx, nk, None, 0.0))  # label=0 (negative)

        # Shuffle samples
        self.rng.shuffle(self.samples)

        pos_count = sum(1 for s in self.samples if s[3] == 1.0)
        neg_count = sum(1 for s in self.samples if s[3] == 0.0)
        print(f"  {split}: {len(self.samples)} samples (pos={pos_count}, neg={neg_count})")
        if self.train:
            print(f"  Aug: noise σ={noise_std}, cutout {cutout_ratio*100:.0f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, img_key, _, label = self.samples[idx]

        # Text: random variant in train, variant 0 in eval
        t = self.text_features[user_idx]
        text_feat = t[self.rng.randint(0, t.size(0) - 1) if self.train else 0].clone()

        # Image: random variant in train, variant 0 in eval
        v = self.image_features[img_key]
        img_feat = v[self.rng.randint(0, v.size(0) - 1) if self.train else 0].clone()

        # Augmentation (train only)
        if self.train:
            # Gaussian noise
            text_feat = text_feat + torch.randn_like(text_feat) * self.noise_std
            img_feat = img_feat + torch.randn_like(img_feat) * self.noise_std

            # Shared Cutout
            if self.cutout_ratio > 0:
                mask = (torch.rand_like(text_feat) > self.cutout_ratio).float()
                text_feat = text_feat * mask
                img_feat = img_feat * mask

        return text_feat, img_feat, torch.tensor(label, dtype=torch.float32)

    def get_user_idx(self, idx):
        return self.samples[idx][0]
