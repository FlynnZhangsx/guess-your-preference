# Personalized Prompt-to-Image Generation via Preference Alignment

**Cross-modal user preference alignment for personalized text-to-image generation.**

Given 0~N reference images + a short text request, the system automatically extracts the user's aesthetic preferences and generates a customized, high-quality image.

## Pipeline Architecture

```
User Images (0~N)  +  Short Text Request
        │                      │
        ▼                      │
CLIP ViT-L/14 (frozen)         │
        │                      │
Image_MLP (trained residual)   │
        │                      │
┌───────┴────────┐             │
│  L2-norm + mean│             │
│  → 768 pref    │             │
└───────┬────────┘             │
        │                      │
        ▼                      │
CLIP Text Encoder               │
(zero-shot style matching)     │
        │                      │
┌───────┴────────┐             │
│ Top-K style    │             │
│ descriptions   │─────────────┘
└───────┬────────┘
        │
        ▼
Qwen3-VL (prompt designer)
        │
        ▼
English Generation Prompt
        │
        ▼
Qwen-Image (DashScope API / local diffusers)
```

## Key Modules

| Module | Model | Description |
|--------|-------|-------------|
| Preference Encoder | `openai/clip-vit-large-patch14` + `Image_MLP` | Frozen CLIP + trained residual MLP (Val AUC=0.8934). 768-dim user preference vector from 1~N images |
| Style Matching | CLIP Text Encoder | Zero-shot cosine similarity matching against 27 predefined aesthetic descriptors (composition, color, lighting, texture, atmosphere) |
| Prompt Enhancement | `Qwen3-VL-4B-Instruct` | Fuses style descriptions + user request → detailed English T2I prompt |
| Image Generation | `qwen-image-2.0-pro` (API) or `Qwen-Image-2512` (local) | Supports DashScope API and local diffusers modes |

## Preference Alignment Model

### Architecture

- **Backbone**: CLIP ViT-L/14 (frozen, 768-dim features)
- **Image_MLP**: Residual projection MLP — Linear(768→384) → LayerNorm → ReLU → Linear(384→768) → residual add → L2 norm
- **Parameters**: ~590K (image_mlp only), negligible inference overhead
- **Key design**: Small-initialized residual connection ($\mathbf{x}' = \mathbf{x} + \Delta(\mathbf{x})$), preserving CLIP pre-training

### Training

- **Data**: Multi-user preference annotations with per-user stratified split (15/15/70 train/val/test)
- **Loss**: BCEWithLogitsLoss with pos_weight for class balance
- **Metrics**: AUC (primary), AP, Accuracy, Hit@K, MRR
- **Best performance**: Val AUC = **0.8934**, AP = 0.8642, Accuracy = 0.8107
- **Training time**: ~5 min on CPU (operating on pre-extracted 768-dim features)

See `train.py` for full training code and `model.py` for architecture details.

## Directory Structure

```
preference_align/
├── model.py                  # PreferenceAlignModel (Residual MLP)
├── train.py                  # Training script (v7 — BCE loss, stratified split)
├── dataset.py                # PreferenceDataset with per-user splits
├── extract_features.py       # Pre-extract CLIP features for training
├── pipeline.py               # Full inference pipeline
├── qwen_bridge.py            # PreferenceProjector (virtual token projection)
├── evaluate_metrics.py       # CLIP-based automated evaluation + ImageGenerationEvaluator
├── run_experiments.py        # Batch experiment runner (Mode A/B/C)
├── api_server.py             # FastAPI backend for Web UI
├── webui.py                  # Gradio-based Web UI (legacy)
├── static/
│   └── index.html            # Midjourney-style Web UI frontend
├── run.sh                    # One-click server launcher
├── data_cache/               # Pre-extracted CLIP features for training
├── experiment_output/        # Generated images from Mode A/B/C experiments
├── output/                   # Runtime generated images
├── human_ratings/            # Human evaluation ratings (JSONL)
├── preference_model_best.pth # Best checkpoint (Val AUC=0.8934)
└── preference_model_v7.pth   # V7 training checkpoint
```

## Quick Start

### 1. Launch Web UI

```bash
# One-click launch (auto-waits for models to load)
bash run.sh

# Or manually
python api_server.py --port 7860
```

Then open `http://localhost:7860` in browser.

### 2. Web UI Usage (Midjourney-style Chat)

1. Upload reference images (optional) via the upload zone
2. Type a short description (e.g., "a cat on a moonlit windowsill")
3. Click **Generate** or press Enter — wait ~30-60s
4. Use the refine bar to iteratively modify: "make it warmer", "add flowers", etc.
5. Click history thumbnails to revisit previous generations

### 3. Python API

```python
from pipeline import PersonalizationPipeline

pipe = PersonalizationPipeline(load_qwen_vl=True)

# Generate personalized prompt
enriched = pipe.generate_personalized_prompt(
    image_paths=["ref1.png", "ref2.png"],
    short_prompt="a summer music festival poster at sunset"
)

# Multi-turn refinement
refined = pipe.refine_prompt_with_feedback(
    current_prompt=enriched,
    user_feedback="make the colors warmer and add more golden light"
)

# Generate image
result = pipe.generate_image_unified(
    prompt=refined, mode="api", size="2048*2048"
)
```

### 4. Run Experiments

```bash
python run_experiments.py          # Run all 16 test cases (Mode A/B/C)
python evaluate_metrics.py         # Compute CLIP metrics
```

### 5. Train Preference Model

```bash
python extract_features.py         # Step 1: Extract CLIP features
python train.py                    # Step 2: Train Image_MLP (~5 min on CPU)
```


## Models

| Model | Source |
|-------|--------|
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` |
| Qwen3-VL-4B-Instruct | Local or `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen-Image (API) | DashScope `qwen-image-2.0-pro` |
| Qwen-Image (Local) | `Qwen/Qwen-Image-2512` via diffusers |
| Image_MLP (ours) | `preference_model_best.pth` (Val AUC=0.8934) |
