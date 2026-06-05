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
    Qwen-Image (local diffusers / DashScope API)

All three modules have training guarantees:
  - CLIP Image/Text Encoder: pretrained by OpenAI
  - Image_MLP (residual): trained by V7 (Val AUC=0.8934)
  - Style matching: zero-shot via cosine similarity in CLIP space
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
DASHSCOPE_API_KEY = "sk-54f1354ff3754d9682f73aea34ad2b47"  # 百炼 API Key

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
                 device: str = DEVICE,
                 load_qwen_vl: Optional[bool] = None):
        self.device = device
        self.qwen_model_name = qwen_model_name
        self.checkpoint_path = checkpoint_path

        # Load the preference encoder lazily. 0-image prompt enhancement does not
        # need CLIP/Image_MLP, so it should not require the preference checkpoint.
        self.encoder = None

        # Load Qwen3-VL when explicitly requested. By default this stays in
        # lightweight/offline simulation mode unless ENABLE_REAL_QWEN_VL=1.
        self.qwen_model = None
        self.qwen_tokenizer = None
        self._load_qwen_vl(qwen_model_name, load_qwen_vl=load_qwen_vl)

    def _load_preference_encoder_if_needed(self):
        if self.encoder is None:
            self.encoder = load_preference_encoder(self.checkpoint_path, self.device)
        return self.encoder

    def _load_qwen_vl(self, model_name: str, load_qwen_vl: Optional[bool] = None):
        """Load Qwen3-VL when enabled by argument or ENABLE_REAL_QWEN_VL=1."""
        if load_qwen_vl is None:
            load_qwen_vl = os.environ.get("ENABLE_REAL_QWEN_VL", "0").lower() in {"1", "true", "yes", "on"}
        if not load_qwen_vl:
            print(f"\n[Qwen-VL] Placeholder: '{model_name}' not loaded.")
            print(f"  Pass load_qwen_vl=True or set ENABLE_REAL_QWEN_VL=1 to enable real inference.")
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor
        import importlib.util
        print(f"\n[Qwen-VL] Loading {model_name}...")
        model_kwargs = {
            "dtype": "auto",
            "trust_remote_code": True,
        }
        if importlib.util.find_spec("accelerate") is not None:
            model_kwargs["device_map"] = "auto"
        self.qwen_model = AutoModelForImageTextToText.from_pretrained(
            model_name, **model_kwargs,
        ).eval()
        if "device_map" not in model_kwargs:
            self.qwen_model = self.qwen_model.to(self.device)
        self.qwen_tokenizer = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.qwen_tokenizer.tokenizer.pad_token is None:
            self.qwen_tokenizer.tokenizer.pad_token = \
                self.qwen_tokenizer.tokenizer.eos_token
        # Qwen3-VL stores hidden_size in text_config sub-config
        cfg = self.qwen_model.config
        hidden = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", "unknown")
        print(f"  Loaded. Hidden size: {hidden}")

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

        # Step 1-2: Extract style profile.
        # When reference images exist → user preference vector via CLIP + Image_MLP.
        # When 0 images → use the centroid of all style descriptors as a neutral
        #   "balanced" preference so the full preference-alignment pipeline still runs.
        if image_paths:
            encoder = self._load_preference_encoder_if_needed()
            pref_vector = extract_user_aesthetic_vector(image_paths, encoder, self.device)
            style_text = match_aesthetic_style(pref_vector, encoder, top_k=5)
        else:
            print("\n[Style Match] No reference images; using balanced centroid preference.")
            encoder = self._load_preference_encoder_if_needed()
            # Neutral preference = centroid of all style embeddings (L2-normalized)
            all_embs = encoder["style_embeddings"]  # [N, 768], already L2-normalized
            neutral = all_embs.mean(dim=0)          # [768]
            neutral = F.normalize(neutral, dim=-1)  # back to unit sphere
            style_text = match_aesthetic_style(neutral, encoder, top_k=5)

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
        """Mock generation for offline testing (when Qwen-VL is not loaded)."""
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
                            size: str = "2048*2048",
                            api_key: str = None,
                            model: str = "qwen-image-2.0-pro",
                            prompt_extend: bool = True,
                            watermark: bool = False) -> dict:
        """
        Call Qwen-Image API via DashScope MultiModalConversation.

        Prerequisites:
            pip install dashscope

        API Key (choose one):
            1. Set env var:  export DASHSCOPE_API_KEY="sk-xxx"
            2. Pass directly: call_qwen_image_api(..., api_key="sk-xxx")
            Get key from: https://help.aliyun.com/zh/model-studio/get-api-key

        Args:
            generated_prompt: English T2I prompt from Qwen3-VL
            output_dir:       where to save the generated image
            negative_prompt:  things to avoid in generation
            size:             output size, e.g. "2048*2048", "1664*928"
            api_key:          DashScope API key (or set DASHSCOPE_API_KEY env var)
            model:            "qwen-image-2.0-pro" or "qwen-image-2.0-plus"
            prompt_extend:    use DashScope's built-in prompt extension
            watermark:        add watermark to output
        Returns:
            dict with keys: success, image_path, image_url, prompt_used, status
        """
        print(f"\n{'='*60}")
        print(f"[Qwen-Image API] Model: {model}")
        print(f"  Prompt: {generated_prompt[:200]}...")
        print(f"  Size: {size}")
        print(f"{'='*60}")

        os.makedirs(output_dir, exist_ok=True)

        # ---- API Key ----
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "") or DASHSCOPE_API_KEY
        if not api_key:
            print(f"  [SKIP] No DASHSCOPE_API_KEY set.")
            print(f"  Set it via: export DASHSCOPE_API_KEY='sk-xxx'")
            print(f"  Or get a key at: https://help.aliyun.com/zh/model-studio/get-api-key")
            return {"success": False, "error": "No API key",
                    "prompt_used": generated_prompt, "status": "no_key"}

        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError:
            print(f"  [ERROR] dashscope not installed. Run: pip install dashscope")
            return {"success": False, "error": "dashscope not installed",
                    "prompt_used": generated_prompt, "status": "no_sdk"}

        # Beijing region URL (use https://dashscope-intl.aliyuncs.com/api/v1 for Singapore)
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

        # Build messages in MultiModalConversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": generated_prompt}
                ]
            }
        ]

        # Default negative prompt (quality guard)
        if not negative_prompt:
            negative_prompt = (
                "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
                "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
            )

        print(f"  Calling DashScope MultiModalConversation...")
        response = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format='message',
            stream=False,
            watermark=watermark,
            prompt_extend=prompt_extend,
            negative_prompt=negative_prompt,
            size=size,
        )

        if response.status_code == 200:
            # Extract image URL from MultiModalConversation response.
            # DashScope returns dict-like objects — use .get() / [] access, not hasattr.
            try:
                output = response.output
                image_url = None

                # Path 1: output.choices[0].message.content[*].image  (result_format='message')
                try:
                    choices = output.get("choices", None) if hasattr(output, "get") else output.choices
                    if choices:
                        msg = choices[0].get("message", {}) if hasattr(choices[0], "get") else choices[0].message
                        content = msg.get("content", []) if hasattr(msg, "get") else msg.content
                        for item in content:
                            url = None
                            if hasattr(item, "get"):
                                url = item.get("image") or item.get("url")
                            else:
                                url = getattr(item, "image", None) or getattr(item, "url", None)
                            if url:
                                image_url = url
                                break
                except Exception:
                    pass

                # Path 2: output.results[0].url  (result_format='url' / ImageSynthesis)
                if not image_url:
                    try:
                        results = output.get("results", None) if hasattr(output, "get") else output.results
                        if results:
                            image_url = results[0].get("url") if hasattr(results[0], "get") else results[0].url
                    except Exception:
                        pass

                if image_url:
                    import urllib.request
                    output_path = os.path.join(
                        output_dir,
                        f"poster_{abs(hash(generated_prompt)) % 100000:05d}.png"
                    )
                    urllib.request.urlretrieve(image_url, output_path)
                    print(f"  Success! Saved to: {output_path}")
                    print(f"  Image URL: {image_url}")
                    print(f"{'='*60}")
                    return {
                        "success": True,
                        "image_path": output_path,
                        "image_url": image_url,
                        "prompt_used": generated_prompt,
                        "status": "generated",
                    }

                # Could not extract URL — dump raw response for debugging
                import json
                raw = json.dumps(response, ensure_ascii=False, default=str)
                print(f"  [WARN] Response OK but could not extract image URL.")
                print(f"  Raw response (first 500 chars): {raw[:500]}")
                return {
                    "success": False,
                    "error": "Could not extract image URL from response",
                    "prompt_used": generated_prompt,
                    "status": "parse_error",
                    "raw_response": raw[:2000],
                }
            except Exception as e:
                import traceback
                print(f"  [ERROR] Failed to parse response: {e}")
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e),
                    "prompt_used": generated_prompt,
                    "status": "parse_error",
                }
        else:
            print(f"  [FAILED] HTTP={response.status_code}, "
                  f"Code={response.code}, Message={response.message}")
            print(f"  参考文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            print(f"{'='*60}")
            return {
                "success": False,
                "error": f"HTTP={response.status_code}, Code={response.code}, {response.message}",
                "prompt_used": generated_prompt,
                "status": "failed",
            }

    # ============================================================
    # 5. Unified Image Generation (local or API)
    # ============================================================

    def generate_image_unified(
        self,
        prompt: str,
        mode: str = "local",
        # ---- shared args ----
        negative_prompt: str = "",
        # ---- local-mode args ----
        output_path: str = "./output/generated.png",
        width: int = 1664,
        height: int = 928,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        qwen_image_model: str = "Qwen/Qwen-Image-2512",
        # ---- api-mode args ----
        output_dir: str = "./generated_posters",
        size: str = "2048*2048",
        api_key: str = None,
        api_model: str = "qwen-image-2.0-pro",
        prompt_extend: bool = True,
        watermark: bool = False,
        **kwargs,
    ) -> dict:
        """
        Unified interface for Qwen-Image generation — local diffusers or DashScope API.

        Usage:
            # Local generation (requires GPU with ~24 GB VRAM)
            result = pipe.generate_image_unified(prompt, mode="local")

            # API generation (requires DASHSCOPE_API_KEY)
            result = pipe.generate_image_unified(prompt, mode="api")

        Args:
            prompt:   English T2I prompt (from Qwen3-VL)
            mode:     "local" (diffusers) or "api" (DashScope MultiModalConversation)

        Local-mode args:
            output_path, width, height, num_inference_steps, true_cfg_scale,
            seed, qwen_image_model

        API-mode args:
            output_dir, size, api_key, api_model, prompt_extend, watermark

        Returns:
            dict with keys: success, image_path, ...
        """
        if mode == "local":
            return self.generate_image(
                prompt=prompt,
                output_path=output_path,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                seed=seed,
                qwen_image_model=qwen_image_model,
            )
        elif mode == "api":
            return self.call_qwen_image_api(
                generated_prompt=prompt,
                output_dir=output_dir,
                negative_prompt=negative_prompt,
                size=size,
                api_key=api_key,
                model=api_model,
                prompt_extend=prompt_extend,
                watermark=watermark,
            )
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'local' or 'api'.")

    # ============================================================
    # 6. GPU Memory Management
    # ============================================================

    def unload_qwen_vl(self):
        """Completely free Qwen3-VL from GPU memory before loading Qwen-Image."""
        print("\n[Unload] Removing Qwen3-VL from GPU...")
        if self.qwen_model is not None:
            del self.qwen_model
            self.qwen_model = None
        if self.qwen_tokenizer is not None:
            del self.qwen_tokenizer
            self.qwen_tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after unload: {mem:.2f} GiB")

    def unload_preference_encoder(self):
        """Free CLIP + Image_MLP from GPU memory before loading Qwen-Image."""
        print("[Unload] Removing preference encoder (CLIP + Image_MLP) from GPU...")
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU memory after unload: {mem:.2f} GiB")

    # ============================================================
    # 7. Qwen-Image Local Generation (via diffusers)
    # ============================================================

    def generate_image(
        self,
        prompt: str,
        output_path: str = "./output/generated.png",
        negative_prompt: str = "",
        width: int = 1664,
        height: int = 928,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        qwen_image_model: str = "Qwen/Qwen-Image-2512",
        quantization: str = "bf16",  # reserved; bf16 only (bitsandbytes unavailable)
    ) -> dict:
        """
        Load Qwen-Image-2512 via diffusers, generate an image, save, and unload.

        IMPORTANT: Call unload_qwen_vl() + unload_preference_encoder() BEFORE this,
        so Qwen-Image has enough GPU VRAM to load on a single 24 GB card.

        Args:
            prompt:               English T2I prompt (from Qwen3-VL)
            output_path:          where to save the generated PNG
            negative_prompt:      quality / artifact avoidance (Chinese or English)
            width, height:        output resolution (default 1664x928 = 16:9)
            num_inference_steps:  diffusion steps (higher = more detail, slower)
            true_cfg_scale:       classifier-free guidance scale
            seed:                 random seed for reproducibility
            qwen_image_model:     HF model id or local path for Qwen-Image
            quantization:         currently only "bf16" supported
        Returns:
            dict with keys: success, image_path, prompt_used, seed
        """
        print(f"\n{'='*60}")
        print(f"[Qwen-Image] Model: {qwen_image_model}")
        print(f"  Prompt: {prompt[:200]}...")
        print(f"  Size: {width}x{height}  Steps: {num_inference_steps}  CFG: {true_cfg_scale}")
        print(f"{'='*60}")

        from diffusers import QwenImagePipeline

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"  GPU allocated before load: {mem:.2f} GiB")

        print(f"  Loading QwenImagePipeline (dtype={torch_dtype})...")
        pipe = QwenImagePipeline.from_pretrained(
            qwen_image_model, torch_dtype=torch_dtype
        ).to(device_str)

        if negative_prompt == "":
            negative_prompt = (
                "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
                "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
            )

        print(f"  Generating image (seed={seed})...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=torch.Generator(device=device_str).manual_seed(seed),
        ).images[0]

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        print(f"  Saved to: {output_path}")

        # Unload
        del pipe
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU after unload: {mem:.2f} GiB")

        print(f"{'='*60}")
        return {"success": True, "image_path": output_path,
                "prompt_used": prompt, "seed": seed}


# ============================================================
# 8. Demo & End-to-End Entry Points
# ============================================================

def run_demo(qwen_image_mode: str = "local", api_key: str = None):
    """Lightweight demo with mocked images. Uses simulated Qwen-VL output by default.

    Args:
        qwen_image_mode: "local" (diffusers) or "api" (DashScope).
        api_key:         DashScope API key (only needed for api mode).
    """
    print("=" * 60)
    print(f"Personalization Pipeline — DEMO  [mode={qwen_image_mode}]")
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

    # Use unified interface
    result = pipeline.generate_image_unified(
        prompt=prompt,
        mode=qwen_image_mode,
        output_dir=str(demo_dir / "output"),
        api_key=api_key,
    )

    print(f"\nDemo result: {result}")

    # Cleanup
    for p in mock_images:
        try: os.remove(p)
        except: pass
    try: demo_dir.rmdir()
    except: pass

    print(f"\n{'='*60}")
    print("Demo complete!")
    print(f"{'='*60}")


def run_end_to_end_0img(
    short_prompt: str = "一只小猫坐在月光下的窗台上",
    qwen_vl_path: str = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct",
    qwen_image_model: str = "Qwen/Qwen-Image-2512",
    output_dir: str = "./output",
    qwen_image_mode: str = "local",   # "local" or "api"
    api_key: str = None,               # needed when qwen_image_mode="api"
):
    """0-image end-to-end: short prompt → Qwen3-VL-4B → unload → Qwen-Image → image.

    Args:
        qwen_image_mode: "local" loads Qwen-Image via diffusers (needs GPU);
                         "api" calls DashScope MultiModalConversation (needs API key).
    """
    print("=" * 60)
    print(f"END-TO-END TEST: 0 Reference Images  [mode={qwen_image_mode}]")
    print("=" * 60)

    pipe = PersonalizationPipeline(qwen_model_name=qwen_vl_path, load_qwen_vl=True)

    enriched = pipe.generate_personalized_prompt(
        image_paths=[], short_prompt=short_prompt,
    )

    # Free everything before image generation (local mode needs VRAM)
    pipe.unload_preference_encoder()
    pipe.unload_qwen_vl()

    out = os.path.join(output_dir, "e2e_0img.png")
    result = pipe.generate_image_unified(
        prompt=enriched,
        mode=qwen_image_mode,
        output_path=out,
        output_dir=output_dir,
        qwen_image_model=qwen_image_model,
        api_key=api_key,
    )

    print(f"\nFinal result: {result}")
    return result


def run_end_to_end_1img(
    short_prompt: str = "一只小猫坐在月光下的窗台上",
    ref_image: str = "/tmp/qwen_demo_ref.jpeg",
    qwen_vl_path: str = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct",
    qwen_image_model: str = "Qwen/Qwen-Image-2512",
    output_dir: str = "./output",
    qwen_image_mode: str = "local",   # "local" or "api"
    api_key: str = None,               # needed when qwen_image_mode="api"
):
    """1-image end-to-end: ref image → preference align → Qwen3-VL → unload → Qwen-Image.

    Args:
        qwen_image_mode: "local" loads Qwen-Image via diffusers (needs GPU);
                         "api" calls DashScope MultiModalConversation (needs API key).
    """
    print("=" * 60)
    print(f"END-TO-END TEST: 1 Reference Image (preference alignment)  [mode={qwen_image_mode}]")
    print("=" * 60)

    pipe = PersonalizationPipeline(qwen_model_name=qwen_vl_path, load_qwen_vl=True)

    enriched = pipe.generate_personalized_prompt(
        image_paths=[ref_image], short_prompt=short_prompt,
    )

    # Free EVERYTHING before loading Qwen-Image (local mode needs VRAM)
    pipe.unload_preference_encoder()
    pipe.unload_qwen_vl()

    out = os.path.join(output_dir, "e2e_1img.png")
    result = pipe.generate_image_unified(
        prompt=enriched,
        mode=qwen_image_mode,
        output_path=out,
        output_dir=output_dir,
        qwen_image_model=qwen_image_model,
        api_key=api_key,
    )

    print(f"\nFinal result: {result}")
    return result


if __name__ == "__main__":
    import sys

    # ---- Parse mode ----
    mode = "local"
    if "--api" in sys.argv:
        mode = "api"
        sys.argv.remove("--api")

    # ---- Parse subcommand ----
    if len(sys.argv) > 1 and sys.argv[1] == "0img":
        run_end_to_end_0img(qwen_image_mode=mode)
    elif len(sys.argv) > 1 and sys.argv[1] == "1img":
        run_end_to_end_1img(qwen_image_mode=mode)
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo(qwen_image_mode=mode)
    else:
        # Default: run 0-image test
        run_end_to_end_0img(qwen_image_mode=mode)
