"""
End-to-End Inference Pipeline (No Extra Modules)
-------------------------------------------------
Personalized poster prompt generation using ONLY trained modules:

    User Images (1~N)  +  Short Text Request
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
    CLIP Text Encoder              │
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
    Qwen-Image API

All three modules have training guarantees:
  - CLIP Image/Text Encoder: pretrained by OpenAI
  - Image_MLP (residual): trained by V7 (Val AUC=0.8934)
  - Style matching: zero-shot via cosine similarity in CLIP space
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from model import PreferenceAlignModel

warnings.filterwarnings("ignore")

# ============================================================
# Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Pipeline] Device: {DEVICE}")

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "preference_model_best.pth"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
QWEN_VL_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# ============================================================
# Aesthetic style descriptors (pre-encoded by CLIP text encoder)
# ============================================================

AESTHETIC_STYLE_DESCRIPTORS = [
    # Composition
    "symmetrical balanced centered composition, classical layout",
    "asymmetrical dynamic off-center composition, modern editorial layout",
    "minimalist negative space, clean geometric arrangement",
    "dense rich maximalist composition, every corner filled with detail",
    # Color palette
    "warm golden amber tones, sunset hues, rich earthy browns",
    "cool oceanic blues, teal, aquamarine, crisp sea foam green",
    "pastel macaron soft candy colors, dreamy confectionery shades",
    "high contrast black and white, stark chiaroscuro, dramatic monochrome",
    "vibrant neon electric, fluorescent brights, cyberpunk glow",
    "muted desaturated tones, foggy misty atmospheric grays",
    "autumnal harvest colors, burnt orange, deep burgundy, golden wheat",
    # Lighting / Mood
    "bright uplifting cheerful mood, radiant warm sunlight, high-key lighting",
    "moody subdued atmospheric shadows, low-key dramatic lighting, film noir",
    "soft diffused natural window light, gentle morning glow, airy atmosphere",
    "golden hour backlight, lens flare, dreamy romantic haze",
    # Style / Texture
    "photorealistic 8K rendering, hyper-detailed, lifelike textures",
    "hand-drawn illustration, watercolor textures, artistic brushstrokes",
    "vector flat design, clean graphic shapes, bold outlines",
    "vintage film grain, analog photography, retro nostalgic patina",
    "futuristic sci-fi, metallic surfaces, holographic iridescence",
    "organic botanical nature motifs, flowing floral patterns, verdant greenery",
    # Atmosphere
    "cozy warm inviting hygge atmosphere, soft blankets, candlelight",
    "luxurious elegant sophisticated, marble surfaces, gold accents",
    "playful whimsical fantastical, floating elements, magical sparkles",
    "urban street style, gritty concrete textures, neon signs at night",
    "serene zen meditation, raked sand, bamboo, tranquil stillness",
    "bold powerful dynamic, explosive energy, dramatic impact",
]


@torch.no_grad()
def _encode_style_descriptors(clip_model, clip_processor, device) -> torch.Tensor:
    """Pre-encode all style descriptors with CLIP text encoder → [N, 768]."""
    texts = list(AESTHETIC_STYLE_DESCRIPTORS)
    all_embs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        inputs = clip_processor(text=batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(device)
        emb = clip_model.get_text_features(**inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output if hasattr(emb, 'pooler_output') and emb.pooler_output is not None \
                  else emb.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)  # [N, 768]


# ============================================================
# 1. Load Preference Encoder
# ============================================================

def load_preference_encoder(checkpoint_path: str = str(CHECKPOINT_PATH),
                            device: str = DEVICE) -> dict:
    """Load CLIP ViT-L/14 + trained Image_MLP."""
    print(f"\n[Encoder] Loading CLIP ViT-L/14 from {CLIP_MODEL_NAME}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    print(f"[Encoder] Loading trained Image_MLP from {checkpoint_path}...")
    pref_model = PreferenceAlignModel(dim=768, hidden_dim=384, dropout=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    pref_model.load_state_dict(state_dict, strict=True)
    pref_model.eval()
    for p in pref_model.parameters():
        p.requires_grad = False

    # Pre-encode style descriptors
    print(f"[Encoder] Pre-encoding {len(AESTHETIC_STYLE_DESCRIPTORS)} style descriptors...")
    style_embeddings = _encode_style_descriptors(clip_model, clip_processor, device)

    print(f"[Encoder] Ready. Image_MLP params: {sum(p.numel() for p in pref_model.image_mlp.parameters()):,}")
    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "image_mlp": pref_model.image_mlp,
        "style_embeddings": style_embeddings,      # [N, 768]
        "style_descriptors": AESTHETIC_STYLE_DESCRIPTORS,  # list of str
    }


# ============================================================
# 2. Extract User Aesthetic Vector
# ============================================================

def extract_user_aesthetic_vector(image_paths: List[str], encoder: dict,
                                  device: str = DEVICE) -> torch.Tensor:
    """1~N images → CLIP + Image_MLP → L2-norm mean → [768] preference vector."""
    if not image_paths:
        raise ValueError("At least one reference image required.")

    clip_model = encoder["clip_model"]
    clip_processor = encoder["clip_processor"]
    image_mlp = encoder["image_mlp"]
    vectors = []

    for i, path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {Path(path).name}")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"    Skip: {e}")
            continue

        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_feat = clip_model.get_image_features(**inputs)
            if not isinstance(clip_feat, torch.Tensor):
                clip_feat = (clip_feat.pooler_output if hasattr(clip_feat, 'pooler_output')
                             and clip_feat.pooler_output is not None
                             else clip_feat.last_hidden_state[:, 0, :])
            clip_feat = F.normalize(clip_feat, dim=-1)
            vec = image_mlp(clip_feat).squeeze(0)  # [768] L2-normed by MLP
            vectors.append(vec)

    if not vectors:
        raise RuntimeError("No valid images to process.")

    stacked = torch.stack(vectors, dim=0)       # [N, 768]
    mean_vec = stacked.mean(dim=0)              # [768]
    pref_vec = F.normalize(mean_vec, dim=-1)    # back to unit sphere

    print(f"  Fused {len(vectors)} images → 768-dim preference vector (norm={pref_vec.norm().item():.4f})")
    return pref_vec


# ============================================================
# 3. Zero-Shot Style Matching
# ============================================================

def match_aesthetic_style(pref_vector: torch.Tensor, encoder: dict,
                          top_k: int = 5) -> str:
    """
    Match preference vector against pre-encoded style descriptors via
    cosine similarity (zero-shot, no training needed).

    Returns a natural language paragraph describing the user's style.
    """
    style_embs = encoder["style_embeddings"].to(pref_vector.device)    # [N, 768]
    descriptors = encoder["style_descriptors"]

    sim = (pref_vector.unsqueeze(0) @ style_embs.T).squeeze(0)        # [N]
    top_indices = sim.argsort(descending=True)[:top_k]

    matched = [descriptors[i] for i in top_indices.tolist()]
    scores = [sim[i].item() for i in top_indices.tolist()]

    print(f"\n[Style Match] Top-{top_k} aesthetic descriptors:")
    for i, (desc, score) in enumerate(zip(matched, scores)):
        print(f"  {i+1}. [{score:.3f}] {desc}")

    # Build style description paragraph
    style_text = (
        f"The user's aesthetic preferences lean toward: "
        f"{matched[0]}; with influences of {matched[1]}; "
        f"and elements of {matched[2]}. "
        f"Overall vibe: {matched[3]}, {matched[4]}."
    )
    return style_text


# ============================================================
# 4. Full Pipeline
# ============================================================

class PersonalizationPipeline:
    """End-to-end personalized poster prompt generation."""

    def __init__(self, checkpoint_path: str = str(CHECKPOINT_PATH),
                 qwen_model_name: str = QWEN_VL_MODEL_NAME,
                 device: str = DEVICE):
        self.device = device
        self.qwen_model_name = qwen_model_name

        # Load trained modules
        self.encoder = load_preference_encoder(checkpoint_path, device)

        # Load Qwen3-VL (placeholder or real)
        self.qwen_model = None
        self.qwen_tokenizer = None
        self._load_qwen_vl(qwen_model_name)

    def _load_qwen_vl(self, model_name: str):
        """Load Qwen3-VL. Set REAL_LOAD=True on GPU server."""
        REAL_LOAD = False
        if not REAL_LOAD:
            print(f"\n[Qwen-VL] Placeholder: '{model_name}' not loaded.")
            print(f"  Set REAL_LOAD=True to enable real inference.")
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"\n[Qwen-VL] Loading {model_name}...")
        self.qwen_model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype="auto", device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.qwen_tokenizer = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.qwen_tokenizer.tokenizer.pad_token is None:
            self.qwen_tokenizer.tokenizer.pad_token = \
                self.qwen_tokenizer.tokenizer.eos_token
        print(f"  Loaded. Hidden: {self.qwen_model.config.hidden_size}")

    def generate_personalized_prompt(self, image_paths: List[str],
                                     short_prompt: str,
                                     max_new_tokens: int = 300) -> str:
        """
        Full pipeline:
          1. Images → preference vector (CLIP + trained Image_MLP)
          2. Preference vector → top-K style text (CLIP zero-shot matching)
          3. Style text + short prompt → Qwen3-VL → detailed English prompt
        """
        print(f"\n{'='*60}")
        print(f"[Generate] {len(image_paths)} ref images, request: \"{short_prompt}\"")
        print(f"{'='*60}")

        # Step 1: Extract preference vector from reference images
        pref_vector = extract_user_aesthetic_vector(image_paths, self.encoder, self.device)

        # Step 2: Match to known style descriptors (zero-shot)
        style_text = match_aesthetic_style(pref_vector, self.encoder, top_k=5)

        # Step 3: Build system prompt with matched style
        system_instruction = (
            "You are an expert Text-to-Image prompt engineer. "
            "Based on analysis of the user's reference images, "
            f"here is their aesthetic profile: {style_text} "
            "Now write ONE detailed, professional English prompt "
            "for an image generation model. Include specifics about "
            "composition, lighting, color palette, style, mood, and artistic quality. "
            "Output ONLY the English prompt, no extra text."
        )

        if self.qwen_model is not None and self.qwen_tokenizer is not None:
            # Real Qwen3-VL inference
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                {"role": "user", "content": [{"type": "text", "text": short_prompt}]},
            ]
            inputs = self.qwen_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )
            input_len = inputs["input_ids"].size(1)
            generated_prompt = self.qwen_tokenizer.decode(
                generated_ids[0][input_len:], skip_special_tokens=True
            ).strip()
        else:
            generated_prompt = self._simulate(short_prompt, style_text)

        print(f"\n[Output] {generated_prompt[:200]}...")
        return generated_prompt

    def _simulate(self, short_prompt: str, style_text: str) -> str:
        """Mock generation for offline testing."""
        return (
            f"A stunning poster design for {short_prompt}, "
            f"incorporating {style_text[:150]}. "
            f"Masterful composition with intentional negative space, "
            f"cinematic lighting with soft volumetric shadows, "
            f"professionally color-graded palette, 8K photorealistic detail, "
            f"elegant typography layout, premium print quality."
        )

    def call_qwen_image_api(self, generated_prompt: str,
                            output_dir: str = "./generated_posters",
                            negative_prompt: str = "",
                            width: int = 1024, height: int = 1024) -> dict:
        """Call Qwen-Image API (placeholder)."""
        print(f"\n{'='*60}")
        print(f"[Qwen-Image API]")
        print(f"  Prompt: {generated_prompt[:200]}...")
        print(f"  Size: {width}x{height}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,
                                   f"poster_{abs(hash(generated_prompt)) % 100000:05d}.png")
        print(f"  [PLACEHOLDER] Would save to: {output_path}")
        print(f"  To enable: use DashScope qwen-image-max API")
        print(f"{'='*60}")
        return {"success": True, "image_path": output_path,
                "prompt_used": generated_prompt, "status": "simulated"}


# ============================================================
# 5. Demo
# ============================================================

def run_demo():
    print("=" * 60)
    print("Personalization Pipeline — DEMO")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        print(f"\n[WARN] Checkpoint not found: {CHECKPOINT_PATH}")
        print("  Place preference_model_best.pth in the directory first.")
        return

    pipeline = PersonalizationPipeline(device=DEVICE)

    # Create mock reference images
    demo_dir = BASE_DIR / "demo_refs"
    demo_dir.mkdir(exist_ok=True)
    mock_images = []
    for i, color in enumerate([(255, 100, 100), (100, 255, 100), (100, 100, 255)]):
        p = demo_dir / f"ref_{i+1}.png"
        if not p.exists():
            Image.new("RGB", (224, 224), color=color).save(p)
        mock_images.append(str(p))

    prompt = pipeline.generate_personalized_prompt(
        image_paths=mock_images,
        short_prompt="a summer music festival poster at sunset on the beach",
    )
    pipeline.call_qwen_image_api(prompt, output_dir=str(demo_dir / "output"))

    # Cleanup
    for p in mock_images:
        try: os.remove(p)
        except: pass
    try: demo_dir.rmdir()
    except: pass

    print(f"\n{'='*60}")
    print("Demo complete! No extra modules needed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_demo()
