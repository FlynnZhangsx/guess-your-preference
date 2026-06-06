"""
Three-Mode Prompt Strategy Comparison — Automated Experiment Runner
====================================================================
Runs three prompt enrichment strategies for each test case and saves
generated images for later evaluation.

Strategies:
  Mode A — Zero-shot Baseline:   short_prompt → Qwen-VL → Qwen-Image
           No preference module. Pure Qwen-VL prompt enrichment.

  Mode B — Hard Prompting:       short_prompt + hardcoded English style text
           → Qwen-VL → Qwen-Image
           Text concatenation baseline. No trained modules at all.
           Hand-written style string, e.g. "minimalist, cool tones, clean lines".

  Mode C — Ours (Preference Align):  ref images → CLIP+Image_MLP (trained)
           → preference vector → top-K style matching → Qwen-VL → Qwen-Image
           Our core contribution: the trained preference alignment pipeline.

Usage:
  # Define test cases in the script, then:
  python run_experiments.py                          # run all test cases, all modes
  python run_experiments.py --cases 0 --mode A       # only case 0, Mode A
  python run_experiments.py --cases 0,1 --mode B,C   # cases 0,1, Modes B & C
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
from PIL import Image

# Allow sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (PersonalizationPipeline, load_preference_encoder,
                       extract_user_aesthetic_vector, match_aesthetic_style,
                       DEVICE, CHECKPOINT_PATH)

# ============================================================
# Config
# ============================================================

QWEN_VL_PATH = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct"

OUTPUT_DIR = Path(__file__).resolve().parent / "experiment_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Test Cases
# ============================================================

@dataclass
class TestCase:
    id: str
    short_prompt: str                           # user's short request (Chinese or English)
    ref_images: List[str] = field(default_factory=list)  # paths to reference images (for Mode C)
    hard_style: str = ""                        # hardcoded style text (for Mode B)
    description: str = ""


# ---- Define your test cases here ----
TEST_CASES: List[TestCase] = [
    TestCase(
        id="cat_moonlight",
        short_prompt="一只小猫坐在月光下的窗台上",
        ref_images=[],   # ← fill with actual reference image paths
        hard_style=(
            "Style: soft dreamy atmosphere, cool silver-blue moonlight tones, "
            "shallow depth of field with gentle bokeh, realistic fur texture, "
            "cozy warm interior contrast with cool night outside, cinematic composition."
        ),
        description="Cat on a moonlit windowsill",
    ),
    TestCase(
        id="summer_poster",
        short_prompt="夏日海滩音乐节海报",
        ref_images=[],
        hard_style=(
            "Style: vibrant tropical colors, warm golden sunset glow, "
            "dynamic asymmetrical composition, bold graphic typography area, "
            "high-energy festival atmosphere, glossy magazine print quality."
        ),
        description="Summer beach music festival poster",
    ),
    TestCase(
        id="city_night",
        short_prompt="赛博朋克风格的未来城市夜景",
        ref_images=[],
        hard_style=(
            "Style: neon-drenched cyberpunk aesthetic, high-contrast lighting, "
            "wet reflective streets, holographic advertisements, deep purple and cyan color palette, "
            "dense urban verticality, cinematic sci-fi atmosphere, rain-slicked surfaces."
        ),
        description="Cyberpunk futuristic city at night",
    ),
]


def load_test_cases(json_path: Optional[str] = None) -> List[TestCase]:
    """Load test cases from Python list or external JSON file."""
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TestCase(**item) for item in data]
    return TEST_CASES


# ============================================================
# Shared Helpers
# ============================================================

def _load_qwen_vl(model_path: str = QWEN_VL_PATH):
    """Load Qwen3-VL model and processor. Returns (model, processor)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import importlib.util

    print(f"  [Load] Qwen3-VL from {model_path}...")
    model_kwargs = {"dtype": "auto", "trust_remote_code": True}
    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs).eval()
    if "device_map" not in model_kwargs:
        model = model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None) or \
             getattr(getattr(cfg, "text_config", None), "hidden_size", "unknown")
    print(f"  [Load] Done. Hidden size: {hidden}")
    return model, processor


def _unload_all(*models):
    """Free GPU memory."""
    for m in models:
        if m is not None:
            del m
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"  [Unload] GPU memory: {mem:.2f} GiB")


def _generate_image_api(prompt: str, output_path: str,
                        size: str = "2048*2048",
                        negative_prompt: str = "") -> dict:
    """Call Qwen-Image API via DashScope."""
    pipe = PersonalizationPipeline(load_qwen_vl=False)
    return pipe.call_qwen_image_api(
        generated_prompt=prompt,
        output_dir=str(Path(output_path).parent),
        size=size,
        negative_prompt=negative_prompt,
        prompt_extend=False,
    )


def _run_qwen_vl_inference(model, processor, system_instruction: str,
                           user_prompt: str, max_new_tokens: int = 300) -> str:
    """Run Qwen-VL chat inference, return decoded response."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
        )
    input_len = inputs["input_ids"].size(1)
    enriched = processor.decode(
        generated_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return enriched


# ============================================================
# Mode A: Zero-shot Baseline
# ============================================================

def run_mode_a(short_prompt: str, output_path: str,
               max_new_tokens: int = 300,
               size: str = "2048*2048",
               negative_prompt: str = "") -> dict:
    """
    Mode A — Zero-shot Baseline.
    short_prompt → Qwen-VL → enriched → Qwen-Image.
    No preference module. No style injection of any kind.
    """
    print(f"\n{'#'*60}")
    print(f"# MODE A — Zero-shot Baseline")
    print(f"# \"{short_prompt}\"")
    print(f"{'#'*60}\n")

    model, processor = _load_qwen_vl(QWEN_VL_PATH)

    system_instruction = (
        "You are an expert Text-to-Image prompt engineer. "
        "Rewrite the user's short request into ONE detailed, professional English prompt "
        "for an image generation model. "
        "Add concrete details about subject, composition, lighting, color palette, style, mood, "
        "camera or rendering quality, and artistic finish while preserving the user's intent. "
        "Output ONLY the English prompt, no extra text."
    )

    enriched = _run_qwen_vl_inference(model, processor, system_instruction,
                                      short_prompt, max_new_tokens)
    print(f"  [Mode A] Enriched ({len(enriched.split())} words): {enriched[:200]}...")

    _unload_all(model, processor)

    result = _generate_image_api(enriched, output_path, size=size,
                                 negative_prompt=negative_prompt)
    return {
        "mode": "A (Zero-shot Baseline)",
        "enriched_prompt": enriched,
        "prompt_words": len(enriched.split()),
        "output_path": output_path,
        "generation": result,
    }


# ============================================================
# Mode B: Hard Prompting (Hand-written Style String)
# ============================================================

def run_mode_b(short_prompt: str, hard_style: str,
               output_path: str,
               max_new_tokens: int = 300,
               size: str = "2048*2048",
               negative_prompt: str = "") -> dict:
    """
    Mode B — Hard Prompting (Text Concatenation Baseline).

    short_prompt + hardcoded English style string → Qwen-VL → enriched → Qwen-Image.
    No trained modules. The style is a hand-written text description.
    This is the simplest form of "prompt engineering" baseline.
    """
    print(f"\n{'#'*60}")
    print(f"# MODE B — Hard Prompting (Hand-written Style)")
    print(f"# \"{short_prompt}\"")
    print(f"# Style: \"{hard_style[:100]}...\"")
    print(f"{'#'*60}\n")

    model, processor = _load_qwen_vl(QWEN_VL_PATH)

    system_instruction = (
        "You are an expert Text-to-Image prompt engineer. "
        "Write ONE detailed, professional English prompt for an image generation model. "
        "Include specifics about composition, lighting, color palette, style, mood, and artistic quality. "
        "Output ONLY the English prompt, no extra text."
    )

    # Concatenate the hard style string into the user prompt
    combined_prompt = f"{short_prompt}\n\nDesired visual style: {hard_style}"

    enriched = _run_qwen_vl_inference(model, processor, system_instruction,
                                      combined_prompt, max_new_tokens)
    print(f"  [Mode B] Enriched ({len(enriched.split())} words): {enriched[:200]}...")

    _unload_all(model, processor)

    result = _generate_image_api(enriched, output_path, size=size,
                                 negative_prompt=negative_prompt)
    return {
        "mode": "B (Hard Prompting)",
        "enriched_prompt": enriched,
        "prompt_words": len(enriched.split()),
        "hard_style_used": hard_style,
        "output_path": output_path,
        "generation": result,
    }


# ============================================================
# Mode C: Ours — Trained Preference Alignment Pipeline
# ============================================================

def run_mode_c(short_prompt: str, ref_image_paths: List[str],
               output_path: str,
               max_new_tokens: int = 300,
               size: str = "2048*2048",
               negative_prompt: str = "") -> dict:
    """
    Mode C — Ours: Trained Preference Alignment Pipeline.

    Pipeline:
      1. ref images → CLIP ViT-L/14 → Image_MLP (trained, Val AUC=0.8934)
         → 768-dim preference vector
      2. Preference vector → top-K cosine match against 27 style descriptors
         → natural language style paragraph
      3. Style paragraph → system prompt → Qwen-VL → enriched prompt
      4. Enriched prompt → Qwen-Image API → image

    This is our core contribution: the fully trained cross-modal preference
    alignment module that learns the user's aesthetic preferences from their
    reference images and injects them as semantically meaningful style text.
    """
    print(f"\n{'#'*60}")
    print(f"# MODE C — Ours (Trained Preference Alignment)")
    print(f"# \"{short_prompt}\", {len(ref_image_paths)} ref images")
    print(f"{'#'*60}\n")

    if not ref_image_paths:
        print("  [Mode C] No reference images. Using neutral centroid preference.")
        # Load encoder just for the style embeddings (centroid)
        encoder = load_preference_encoder(str(CHECKPOINT_PATH), DEVICE)
        import torch.nn.functional as F
        all_embs = encoder["style_embeddings"]
        pref_vector = F.normalize(all_embs.mean(dim=0), dim=-1)
        style_text = match_aesthetic_style(pref_vector, encoder, top_k=5)
    else:
        print("  [Mode C] Loading trained preference encoder...")
        encoder = load_preference_encoder(str(CHECKPOINT_PATH), DEVICE)
        print("  [Mode C] Extracting user preference vector...")
        pref_vector = extract_user_aesthetic_vector(ref_image_paths, encoder, DEVICE)
        print("  [Mode C] Matching aesthetic styles...")
        style_text = match_aesthetic_style(pref_vector, encoder, top_k=5)

    print(f"  [Mode C] Style profile: {style_text[:200]}...")

    # Free encoder BEFORE loading Qwen-VL (24GB card can't hold both)
    del encoder
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load Qwen-VL with the style-injected system prompt
    model, processor = _load_qwen_vl(QWEN_VL_PATH)

    system_instruction = (
        "You are an expert Text-to-Image prompt engineer. "
        "Based on analysis of the user's reference images, "
        f"here is their aesthetic profile: {style_text} "
        "Now write ONE detailed, professional English prompt "
        "for an image generation model. Include specifics about "
        "composition, lighting, color palette, style, mood, and artistic quality. "
        "Output ONLY the English prompt, no extra text."
    )

    enriched = _run_qwen_vl_inference(model, processor, system_instruction,
                                      short_prompt, max_new_tokens)
    print(f"  [Mode C] Enriched ({len(enriched.split())} words): {enriched[:200]}...")

    _unload_all(model, processor)

    result = _generate_image_api(enriched, output_path, size=size,
                                 negative_prompt=negative_prompt)
    return {
        "mode": "C (Ours: Preference Alignment)",
        "enriched_prompt": enriched,
        "prompt_words": len(enriched.split()),
        "style_text": style_text,
        "num_ref_images": len(ref_image_paths),
        "output_path": output_path,
        "generation": result,
    }


# ============================================================
# Experiment Runner
# ============================================================

def run_experiments(test_cases: List[TestCase],
                    modes: List[str] = None,
                    case_indices: List[int] = None,
                    size: str = "2048*2048") -> Dict:
    """Run all test cases through all specified modes."""

    if modes is None:
        modes = ["A", "B", "C"]
    if case_indices is not None:
        test_cases = [test_cases[i] for i in case_indices if i < len(test_cases)]

    all_results = {}

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"  TEST CASE [{case.id}] \"{case.short_prompt}\"")
        print(f"  Ref images: {len(case.ref_images)}  |  {case.description}")
        print(f"{'='*70}")

        case_dir = OUTPUT_DIR / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_results = {
            "id": case.id,
            "short_prompt": case.short_prompt,
            "ref_images": case.ref_images,
            "hard_style": case.hard_style,
            "description": case.description,
        }

        # ---- Mode A ----
        if "A" in modes:
            out_path = str(case_dir / f"{case.id}_mode_A.png")
            try:
                case_results["mode_A"] = run_mode_a(
                    case.short_prompt, out_path, size=size)
            except Exception as e:
                print(f"  [ERROR] Mode A: {e}")
                import traceback
                traceback.print_exc()
                case_results["mode_A"] = {"error": str(e)}

        # ---- Mode B ----
        if "B" in modes:
            out_path = str(case_dir / f"{case.id}_mode_B.png")
            try:
                case_results["mode_B"] = run_mode_b(
                    case.short_prompt, case.hard_style, out_path, size=size)
            except Exception as e:
                print(f"  [ERROR] Mode B: {e}")
                import traceback
                traceback.print_exc()
                case_results["mode_B"] = {"error": str(e)}

        # ---- Mode C ----
        if "C" in modes:
            out_path = str(case_dir / f"{case.id}_mode_C.png")
            try:
                case_results["mode_C"] = run_mode_c(
                    case.short_prompt, case.ref_images, out_path, size=size)
            except Exception as e:
                print(f"  [ERROR] Mode C: {e}")
                import traceback
                traceback.print_exc()
                case_results["mode_C"] = {"error": str(e)}

        all_results[case.id] = case_results

        # Save after each case (resilience against mid-run failures)
        report_path = OUTPUT_DIR / "experiment_results.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  DONE — {len(test_cases)} cases × {len(modes)} modes")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")
    return all_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-mode prompt strategy comparison experiment")
    parser.add_argument("--cases", type=str, default=None,
                        help="Case indices, e.g. '0,1'. Default: all.")
    parser.add_argument("--mode", type=str, default=None,
                        help="Modes: A, B, C or comma-separated. Default: all.")
    parser.add_argument("--test-cases-json", type=str, default=None,
                        help="JSON file with test cases.")
    parser.add_argument("--size", type=str, default="2048*2048",
                        help="Output image size.")
    args = parser.parse_args()

    test_cases = load_test_cases(args.test_cases_json)
    if not test_cases:
        print("ERROR: No test cases. Edit TEST_CASES or use --test-cases-json.")
        return

    modes = [m.strip() for m in args.mode.split(",")] if args.mode else None
    case_indices = [int(i.strip()) for i in args.cases.split(",")] if args.cases else None

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {len(test_cases)} cases × {len(modes or ['A','B','C'])} modes")
    for tc in test_cases:
        print(f"    [{tc.id}] {tc.short_prompt}  (refs={len(tc.ref_images)}, "
              f"style={len(tc.hard_style)} chars)")
    print(f"{'='*70}")

    results = run_experiments(test_cases, modes, case_indices, args.size)

    # Quick summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for case_id, cd in results.items():
        print(f"\n  [{case_id}] \"{cd['short_prompt']}\"")
        for mk in ["mode_A", "mode_B", "mode_C"]:
            if mk not in cd:
                continue
            m = cd[mk]
            if "error" in m:
                print(f"    {mk}: ❌ {m['error']}")
            else:
                p = m.get("generation", {})
                ok = "✅" if p.get("success") else "❌"
                print(f"    {mk}: {ok}  {m['prompt_words']} words  "
                      f"→ {m.get('output_path', 'N/A')}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
