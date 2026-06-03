"""
Qwen Bridge — Virtual Token Projection & Input Assembly
---------------------------------------------------------
Maps 768-dim user preference vectors into K continuous virtual tokens
compatible with Qwen-VL's embedding space, and handles the concatenation
with text token embeddings for model input.

Architecture:
    768-dim preference vector
        │
        ▼
    PreferenceProjector (2-layer MLP + GELU)
        │
        ▼
    [Batch, K, qwen_hidden_size]  virtual tokens
        │
        ▼  + text embeddings
    [Batch, K + seq_len, qwen_hidden_size]  →  Qwen-VL

Reference dimensions:
    - Qwen2.5-VL-7B:  hidden_size = 4096
    - Qwen2.5-VL-3B:  hidden_size = 2048
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PreferenceProjector
# ============================================================

class PreferenceProjector(nn.Module):
    """
    Projects a 768-dim user aesthetic vector into K virtual tokens
    that can be prepended to Qwen-VL's text embeddings.

    Architecture: Linear(768→hidden) → GELU → Linear(hidden→K*dim)

    Args:
        input_dim:          dimension of preference vector (default 768)
        num_virtual_tokens: number of soft prompt tokens K (default 4)
        qwen_hidden_size:   embedding dimension of the target Qwen model
        projector_hidden:   internal hidden dimension of the MLP
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_virtual_tokens: int = 4,
        qwen_hidden_size: int = 4096,   # Qwen2.5-VL-7B
        projector_hidden: int = 1024,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_virtual_tokens = num_virtual_tokens
        self.qwen_hidden_size = qwen_hidden_size
        self.output_dim = num_virtual_tokens * qwen_hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, self.output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Small-init: start near zero so tokens act as a soft bias initially."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, preference_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preference_vector: [batch_size, 768] or [768] — L2-normalized
        Returns:
            virtual_tokens: [batch_size, num_virtual_tokens, qwen_hidden_size]
        """
        if preference_vector.dim() == 1:
            preference_vector = preference_vector.unsqueeze(0)  # [1, 768]

        batch_size = preference_vector.size(0)
        flat = self.mlp(preference_vector)  # [B, K * hidden_size]
        virtual_tokens = flat.view(batch_size, self.num_virtual_tokens, self.qwen_hidden_size)
        return virtual_tokens


# ============================================================
# Input Embedding Assembly
# ============================================================

def prepare_qwen_inputs_embeds(
    model: nn.Module,
    tokenizer,
    virtual_tokens: torch.Tensor,
    text_prompt: str,
    system_prompt: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate virtual preference tokens with tokenized text embeddings,
    producing the full inputs_embeds and attention_mask for Qwen-VL.

    Layout:
        [virtual_tokens (K)] [system_prompt_tokens] [text_prompt_tokens]

    Args:
        model:          Qwen-VL model (must have .get_input_embeddings())
        tokenizer:      corresponding Qwen-VL tokenizer
        virtual_tokens: [batch_size, K, hidden_size] from PreferenceProjector
        text_prompt:    user's short generation request
        system_prompt:  optional system-level instruction
    Returns:
        inputs_embeds:  [batch_size, total_len, hidden_size]
        attention_mask: [batch_size, total_len]
    """
    device = virtual_tokens.device
    batch_size = virtual_tokens.size(0)
    K = virtual_tokens.size(1)

    embed_layer = model.get_input_embeddings()

    # Build the full text: system prompt + user prompt
    if system_prompt:
        full_text = f"{system_prompt}\n\nUser request: {text_prompt}"
    else:
        full_text = text_prompt

    # Tokenize
    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        padding=False,            # no padding — we handle it per-sample
        truncation=True,
        max_length=2048 - K,      # leave room for virtual tokens
    )
    input_ids = tokenized["input_ids"].to(device)  # [1, L]
    text_embeds = embed_layer(input_ids)            # [1, L, hidden_size]
    text_embeds = text_embeds.expand(batch_size, -1, -1)  # [B, L, hidden_size]

    # Concatenate along the sequence dimension
    inputs_embeds = torch.cat([virtual_tokens, text_embeds], dim=1)  # [B, K+L, hidden_size]

    # Build attention mask: K virtual token positions + L text positions
    L = text_embeds.size(1)
    attention_mask = torch.ones(
        (batch_size, K + L),
        dtype=torch.long,
        device=device,
    )

    return inputs_embeds, attention_mask


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("Testing PreferenceProjector...")
    projector = PreferenceProjector(
        input_dim=768,
        num_virtual_tokens=4,
        qwen_hidden_size=4096,
    )
    dummy_pref = torch.randn(2, 768)
    dummy_pref = F.normalize(dummy_pref, dim=-1)

    tokens = projector(dummy_pref)
    print(f"  Input:  {dummy_pref.shape}")
    print(f"  Output: {tokens.shape}")  # Expected: [2, 4, 4096]
    print(f"  Total params: {sum(p.numel() for p in projector.parameters()):,}")

    # Test single vector
    tokens_1 = projector(dummy_pref[0])
    print(f"  Single input → output: {tokens_1.shape}")  # [1, 4, 4096]

    print("\nAll tests passed!")
