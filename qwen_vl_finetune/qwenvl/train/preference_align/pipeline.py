"""
End-to-End Inference Pipeline
-------------------------------
Personalized poster prompt generation pipeline:

    User Images (1~N)  +  Short Text Request
            │                      │
            ▼                      │
    CLIP ViT-L/14 (frozen)         │
            │                      │
    Image_MLP (residual, trained)  │
            │                      │
    ┌───────┴────────┐             │
    │  L2 norm + mean │             │
    │  → 768-dim pref │             │
    └───────┬────────┘             │
            ▼                      │
    PreferenceProjector             │
            │                      │
    [K virtual tokens]             │
            │                      │
            └──────────────────────┘
                    │
                    ▼
            Qwen-VL (prompt designer)
                    │
                    ▼
            English Generation Prompt
                    │
                    ▼
            Qwen-Image API (image generator)

Usage (future server deployment):
    python pipeline.py --images ./refs/ --prompt "summer music festival poster"
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Local imports
from model import PreferenceAlignModel
from qwen_bridge import PreferenceProjector, prepare_qwen_inputs_embeds

warnings.filterwarnings("ignore")

# ============================================================
# Device Selection
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Pipeline] Device: {DEVICE}")

# ============================================================
# Model Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "preference_model_best.pth"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Qwen model placeholder (set to actual path or HF repo when deploying)
QWEN_VL_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"    # 8B is practical for deployment
# For larger capacity: "Qwen/Qwen3-VL-235B-A22B-Instruct" (MoE, ~22B active)
QWEN_HIDDEN_SIZE = 4096           # Qwen3-VL-8B hidden_size
NUM_VIRTUAL_TOKENS = 4
PROJECTOR_HIDDEN = 1024

# ============================================================
# 1. Load Preference Encoder (CLIP + trained Image_MLP)
# ============================================================

def load_preference_encoder(
    checkpoint_path: str,
    device: str = DEVICE,
) -> dict:
    """
    Load CLIP ViT-L/14 and the trained residual Image_MLP from checkpoint.

    Args:
        checkpoint_path: path to preference_model_best.pth
        device:          "cuda" or "cpu"
    Returns:
        dict with keys:
            "clip_model":     CLIPModel (ViT-L/14)
            "clip_processor": CLIPProcessor
            "image_mlp":      ResidualProjectionMLP (trained)
            "model":          PreferenceAlignModel (full, for reference)
    """
    print(f"\n[Encoder] Loading CLIP ViT-L/14 from {CLIP_MODEL_NAME}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad = False

    print(f"[Encoder] Loading trained Image_MLP from {checkpoint_path}...")
    # Instantiate the full PreferenceAlignModel to match checkpoint structure
    pref_model = PreferenceAlignModel(dim=768, hidden_dim=384, dropout=0.0).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    pref_model.load_state_dict(state_dict, strict=True)
    pref_model.eval()

    for param in pref_model.parameters():
        param.requires_grad = False

    # Extract just the image MLP for convenience
    image_mlp = pref_model.image_mlp

    print(f"[Encoder] Loaded. Image_MLP params: {sum(p.numel() for p in image_mlp.parameters()):,}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Checkpoint val_acc: {ckpt.get('val_acc', 'N/A')}")

    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "image_mlp": image_mlp,
        "model": pref_model,
    }


# ============================================================
# 2. Extract User Aesthetic Vector from Reference Images
# ============================================================

def extract_user_aesthetic_vector(
    image_paths: List[str],
    encoder: dict,
    device: str = DEVICE,
) -> torch.Tensor:
    """
    Process 1~N user-uploaded reference images through CLIP + Image_MLP,
    L2-normalize each result, then average to produce a single 768-dim
    preference vector representing the user's implicit aesthetic style.

    Multi-image fusion strategy:
        1. For each image: CLIP encode → 768-dim L2-normed feature
        2. Pass through trained Image_MLP (residual) → 768-dim L2-normed
        3. Mean-pool all normalized vectors
        4. L2-normalize the mean → final 768-dim vector

    This ensures:
        - Each image contributes equally (L2 norm = 1 before averaging)
        - The final vector stays on the unit hypersphere
        - Outliers are naturally suppressed by mean pooling

    Args:
        image_paths: list of paths to user-uploaded reference images
        encoder:     dict from load_preference_encoder()
        device:      "cuda" or "cpu"
    Returns:
        preference_vector: [768] L2-normalized tensor on `device`
    """
    if not image_paths:
        raise ValueError("At least one reference image is required.")

    clip_model = encoder["clip_model"]
    clip_processor = encoder["clip_processor"]
    image_mlp = encoder["image_mlp"]

    aesthetic_vectors = []

    for i, path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] Processing: {path}")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"    Warning: Failed to load {path}: {e}. Skipping.")
            continue

        # CLIP encode
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_feat = clip_model.get_image_features(**inputs)

            # Handle transformers API variance
            if not isinstance(clip_feat, torch.Tensor):
                clip_feat = (
                    clip_feat.pooler_output
                    if hasattr(clip_feat, "pooler_output") and clip_feat.pooler_output is not None
                    else clip_feat.last_hidden_state[:, 0, :]
                )

            clip_feat = F.normalize(clip_feat, dim=-1)        # L2 norm

            # Image_MLP (residual: learns a preference delta)
            aesthetic_vec = image_mlp(clip_feat)               # [1, 768], already L2-normed
            aesthetic_vectors.append(aesthetic_vec.squeeze(0))

    if not aesthetic_vectors:
        raise RuntimeError("No valid images could be processed.")

    # Mean-pool and re-normalize
    stacked = torch.stack(aesthetic_vectors, dim=0)            # [N, 768]
    mean_vec = stacked.mean(dim=0)                             # [768]
    preference_vector = F.normalize(mean_vec, dim=-1)          # L2 unit

    print(f"  Fused {len(aesthetic_vectors)} images → 768-dim preference vector")
    print(f"  Norm: {preference_vector.norm().item():.4f}")
    return preference_vector


# ============================================================
# 3. Full Inference Pipeline
# ============================================================

class PersonalizationPipeline:
    """
    Complete personalized poster generation pipeline.

    Usage:
        pipeline = PersonalizationPipeline(checkpoint_path=..., device=...)
        prompt = pipeline.generate_personalized_prompt(
            image_paths=["ref1.jpg", "ref2.jpg"],
            short_prompt="summer music festival poster",
        )
        result = pipeline.call_qwen_image_api(prompt)
    """

    def __init__(
        self,
        checkpoint_path: str = str(CHECKPOINT_PATH),
        qwen_model_name: str = QWEN_VL_MODEL_NAME,
        device: str = DEVICE,
        num_virtual_tokens: int = NUM_VIRTUAL_TOKENS,
        qwen_hidden_size: int = QWEN_HIDDEN_SIZE,
    ):
        self.device = device
        self.num_virtual_tokens = num_virtual_tokens
        self.qwen_hidden_size = qwen_hidden_size

        # --- Phase 1: Load preference encoder ---
        self.encoder = load_preference_encoder(checkpoint_path, device)

        # --- Phase 2: Load PreferenceProjector ---
        print(f"\n[Bridge] Initializing PreferenceProjector...")
        self.projector = PreferenceProjector(
            input_dim=768,
            num_virtual_tokens=num_virtual_tokens,
            qwen_hidden_size=qwen_hidden_size,
            projector_hidden=PROJECTOR_HIDDEN,
        ).to(device)
        print(f"  Projector params: {sum(p.numel() for p in self.projector.parameters()):,}")

        # --- Phase 3: Load Qwen-VL (placeholder) ---
        print(f"\n[Qwen-VL] Would load: {qwen_model_name}")
        print(f"  Hidden size: {qwen_hidden_size}")
        self.qwen_model = None
        self.qwen_tokenizer = None
        self._load_qwen_vl_placeholder(qwen_model_name)

    def _load_qwen_vl_placeholder(self, model_name: str):
        """
        Load Qwen3-VL model for personalized prompt generation.

        Key design: we inject virtual tokens directly as inputs_embeds,
        bypassing the vision encoder. No pixel_values/image tokens needed.

        Set REAL_LOAD=True below to enable actual model loading.
        """
        # ============================================================
        # Set to True when running on GPU server
        # ============================================================
        REAL_LOAD = False

        if not REAL_LOAD:
            print(f"  [PLACEHOLDER] Qwen3-VL model '{model_name}' not loaded.")
            print(f"  Set REAL_LOAD=True in pipeline.py to enable real inference.")
            return

        # ============================================================
        # Real loading — Qwen3-VL API
        # ============================================================
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"  Loading {model_name} on {self.device}...")

        # --- Model ---
        # dtype="auto" lets the model pick optimal dtype per layer
        self.qwen_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        ).eval()

        # --- Processor (tokenizer + image processor) ---
        # We use AutoProcessor for the chat template and tokenizer.
        # The image processor part is NOT used (virtual tokens replace images).
        self.qwen_tokenizer = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set (needed for attention mask)
        if self.qwen_tokenizer.tokenizer.pad_token is None:
            self.qwen_tokenizer.tokenizer.pad_token = \
                self.qwen_tokenizer.tokenizer.eos_token

        print(f"  Loaded. Vocab: {self.qwen_tokenizer.tokenizer.vocab_size}, "
              f"Hidden: {self.qwen_model.config.hidden_size}")

    def generate_personalized_prompt(
        self,
        image_paths: List[str],
        short_prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate a high-quality, personalized English Image Generation prompt.

        Pipeline:
            1. Extract 768-dim preference vector from reference images
            2. Project to K virtual tokens via PreferenceProjector
            3. Assemble system prompt + virtual tokens + user request
            4. Feed to Qwen-VL, generate detailed English prompt

        Args:
            image_paths:  paths to 1~N user reference images
            short_prompt: user's brief request (e.g., "summer music festival")
            max_new_tokens: max tokens for Qwen-VL to generate
        Returns:
            generated_prompt: detailed English Text-to-Image prompt
        """
        print(f"\n{'='*60}")
        print(f"[Generate] Personalized Prompt Generation")
        print(f"{'='*60}")
        print(f"  Short request: \"{short_prompt}\"")
        print(f"  Reference images: {len(image_paths)}")

        # Step 1: Extract preference vector
        pref_vector = extract_user_aesthetic_vector(
            image_paths, self.encoder, self.device
        )  # [768]

        # Step 2: Project to virtual tokens
        print(f"\n[Projector] 768-dim → {self.num_virtual_tokens} virtual tokens ({self.qwen_hidden_size}-dim)")
        virtual_tokens = self.projector(pref_vector)  # [1, K, hidden_size]

        # Step 3: Assemble system instructions via chat template
        system_instruction = (
            "You are an expert aesthetic designer for Text-to-Image generation. "
            "Virtual aesthetic preference tokens from the user's reference images "
            "have been prepended to your input. "
            "Based on the user's request below, write ONE detailed, professional "
            "English prompt suitable for a high-quality image generation model. "
            "Include: composition, lighting, color palette, style, atmosphere, "
            "artistic references. Output ONLY the English prompt."
        )

        if self.qwen_model is not None and self.qwen_tokenizer is not None:
            # Real inference with Qwen3-VL

            # Step 3a: Build chat messages (text-only, no image)
            # Virtual tokens are prepended to embeddings, not in the chat template
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": short_prompt}],
                },
            ]

            # Step 3b: Tokenize text via chat template
            text_inputs = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            # Step 3c: Get text embeddings
            embed_layer = self.qwen_model.get_input_embeddings()
            text_embeds = embed_layer(text_inputs["input_ids"])      # [1, L, hidden]
            text_mask = text_inputs["attention_mask"]                # [1, L]

            # Step 3d: Prepend virtual tokens
            inputs_embeds = torch.cat([virtual_tokens, text_embeds], dim=1)  # [1, K+L, H]
            attn_mask = torch.cat([
                torch.ones(1, virtual_tokens.size(1), dtype=torch.long, device=self.device),
                text_mask,
            ], dim=1)  # [1, K+L]

            # Step 3e: Generate
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Decode (skip the input portion, only keep generated tokens)
            input_len = inputs_embeds.size(1)
            generated_ids = outputs[0][input_len:]
            generated_prompt = self.qwen_tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
        else:
            # Placeholder: simulate what Qwen-VL would generate
            generated_prompt = self._simulate_generation(
                short_prompt, pref_vector
            )

        print(f"\n[Output] Generated Prompt:")
        print(f"  {generated_prompt[:300]}...")

        return generated_prompt

    def _simulate_generation(
        self,
        short_prompt: str,
        pref_vector: torch.Tensor,
    ) -> str:
        """
        Simulate Qwen-VL output for offline testing.
        In production, this is replaced by real model.generate().
        """
        # Build a plausible mock prompt demonstrating what the system would produce
        mock_prompt = (
            f"A visually stunning poster design for {short_prompt}, "
            f"featuring a balanced composition with harmonious color grading, "
            f"soft volumetric lighting with subtle rim lights, "
            f"8K ultra-high resolution, photorealistic rendering, "
            f"professional graphic design layout with elegant typography spacing, "
            f"cinematic depth of field, premium print quality, "
            f"vibrant yet sophisticated color palette tailored to the user's aesthetic profile, "
            f"clean vector-style decorative elements, masterpiece-level artistry."
        )
        return mock_prompt

    def call_qwen_image_api(
        self,
        generated_prompt: str,
        output_dir: str = "./generated_posters",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
    ) -> dict:
        """
        Call Qwen-Image API to generate the actual poster image.

        This is an API placeholder. In production, replace with:
            - DashScope API (qwen-image-plus / qwen-image-max)
            - Or any compatible Text-to-Image API

        Args:
            generated_prompt: English prompt from generate_personalized_prompt()
            output_dir:       directory to save generated images
            negative_prompt:  optional negative prompt
            width, height:    output image dimensions
        Returns:
            dict with keys: "success", "image_path", "prompt_used"
        """
        print(f"\n{'='*60}")
        print(f"[Qwen-Image API] Generating poster...")
        print(f"{'='*60}")
        print(f"  Prompt: {generated_prompt[:200]}...")
        print(f"  Negative: {negative_prompt or '(none)'}")
        print(f"  Size: {width}x{height}")

        os.makedirs(output_dir, exist_ok=True)

        # ================================================================
        # PLACEHOLDER: Replace with actual API call in production
        # ================================================================
        # Example using DashScope (uncomment and install dashscope):
        #
        # import dashscope
        # from dashscope import ImageSynthesis
        #
        # response = ImageSynthesis.call(
        #     model="qwen-image-max",
        #     prompt=generated_prompt,
        #     negative_prompt=negative_prompt,
        #     n=1,
        #     size=f"{width}*{height}",
        # )
        #
        # if response.status_code == 200:
        #     # Download and save the image from response.output.results[0].url
        #     ...

        # For now, simulate success
        output_path = os.path.join(
            output_dir,
            f"poster_{abs(hash(generated_prompt)) % 100000:05d}.png",
        )
        print(f"  [SIMULATED] Image would be saved to: {output_path}")
        print(f"  [SIMULATED] API call successful! (placeholder)")
        print(f"{'='*60}")

        return {
            "success": True,
            "image_path": output_path,
            "prompt_used": generated_prompt,
            "status": "simulated",
        }


# ============================================================
# 4. Test / Demo
# ============================================================

def run_demo():
    """
    End-to-end demo with mock data.
    Creates dummy reference images and runs the full pipeline.
    """
    print("=" * 60)
    print("Personalization Pipeline — DEMO")
    print("=" * 60)

    # Create mock reference images
    demo_dir = Path(BASE_DIR) / "demo_refs"
    demo_dir.mkdir(exist_ok=True)

    mock_images = []
    for i in range(3):
        img_path = demo_dir / f"ref_{i+1}.png"
        if not img_path.exists():
            # Create a simple colored placeholder image
            color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][i]
            img = Image.new("RGB", (224, 224), color=color)
            img.save(img_path)
        mock_images.append(str(img_path))

    # Check if checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"\n[WARNING] Checkpoint not found: {CHECKPOINT_PATH}")
        print("  Running in MOCK mode without real encoder.")
        print("  Train the model first, or place preference_model_best.pth in the directory.")
        print("  Demo will skip encoder-dependent steps.\n")
        return

    # Initialize pipeline
    pipeline = PersonalizationPipeline(
        checkpoint_path=str(CHECKPOINT_PATH),
        device=DEVICE,
    )

    # Generate personalized prompt
    generated_prompt = pipeline.generate_personalized_prompt(
        image_paths=mock_images,
        short_prompt="a cozy autumn cafe poster with warm lighting",
    )

    # Simulate image generation
    result = pipeline.call_qwen_image_api(
        generated_prompt=generated_prompt,
        output_dir=str(demo_dir / "output"),
    )

    print(f"\n{'='*60}")
    print("Demo complete! Pipeline summary:")
    print(f"  Input images:   {len(mock_images)}")
    print(f"  Short prompt:   \"a cozy autumn cafe poster with warm lighting\"")
    print(f"  Generated:      {generated_prompt[:150]}...")
    print(f"  API result:     {result['status']}")
    print(f"{'='*60}")

    # Cleanup mock images
    for p in mock_images:
        try:
            os.remove(p)
        except:
            pass
    try:
        demo_dir.rmdir()
    except:
        pass


if __name__ == "__main__":
    run_demo()
