"""
Preference Alignment Model (Residual MLP v4)
----------------------------------------------
Key innovation: Residual connection preserves CLIP pre-training.

Architecture:
    x (512-dim, L2-normed CLIP feature)
      ↓
    MLP: Linear(512→256) → LayerNorm → ReLU → Dropout
         → Linear(256→512)
      ↓
    out = x + MLP(x)   ← Residual: model learns a "delta" from CLIP space
      ↓
    F.normalize(out)   ← back to unit hypersphere

At initialization (small weights): delta ≈ 0, so out ≈ original CLIP feature.
This means the model starts FROM CLIP's alignment, then learns to adjust.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualProjectionMLP(nn.Module):
    """
    Residual MLP: learns a preference-specific offset in CLIP feature space.

    Shape: 512 (in) → 256 (hidden) → 512 (out, residual add)
    """

    def __init__(self, dim=768, hidden_dim=768, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, dim)

        # Initialize small so residual starts near identity
        self._init_weights()

    def _init_weights(self):
        # fc1: Xavier uniform (standard)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # fc2: near-zero init so delta ≈ 0 at start
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Args:
            x: [*, 512] L2-normalized CLIP features
        Returns:
            [*, 512] L2-normalized preference-adjusted features
        """
        delta = self.fc1(x)
        delta = self.ln1(delta)
        delta = F.relu(delta)
        delta = self.dropout(delta)
        delta = self.fc2(delta)

        # Residual: CLIP feature + learned preference offset
        out = x + delta
        out = F.normalize(out, p=2, dim=-1)
        return out


class PreferenceAlignModel(nn.Module):
    """
    Cross-modal preference alignment with residual MLP heads.

    Both text and image MLPs operate at native CLIP dimension (512),
    learning preference-specific offsets via residual connections.
    """

    def __init__(self, dim=768, hidden_dim=768, dropout=0.2):
        super().__init__()
        self.text_mlp = ResidualProjectionMLP(dim, hidden_dim, dropout)
        self.image_mlp = ResidualProjectionMLP(dim, hidden_dim, dropout)

    def encode_text(self, text_features):
        """
        Args:
            text_features: [B, 512] pre-extracted CLIP text features (L2-normed)
        Returns:
            [B, 512] preference-adjusted text vectors
        """
        return self.text_mlp(text_features)

    def encode_image(self, image_features):
        """
        Args:
            image_features: [B, 512] or [B, K, 512]
        Returns:
            [B, 512] or [B, K, 512] preference-adjusted image vectors
        """
        if image_features.dim() == 3:
            B, K, D = image_features.shape
            flat = image_features.reshape(B * K, D)
            proj_flat = self.image_mlp(flat)
            return proj_flat.reshape(B, K, -1)
        else:
            return self.image_mlp(image_features)

    def forward(self, text_features, image_features):
        return self.encode_image(image_features)

    def compute_similarity(self, text_features, image_features):
        text_proj = self.encode_text(text_features)
        img_proj = self.encode_image(image_features)
        if img_proj.dim() == 3:
            return (text_proj.unsqueeze(1) * img_proj).sum(dim=-1)
        else:
            return (text_proj * img_proj).sum(dim=-1)
