"""
Three-Mode Prompt Strategy Comparison — Automated Experiment Runner
====================================================================
Runs three prompt enrichment strategies for each test case and saves
generated images for later evaluation.

Strategies:
  Mode A — Zero-shot Baseline: short_prompt → Qwen-VL → Qwen-Image
           (No preference module. Pure Qwen-VL prompt enrichment.)
  Mode B — Hard Prompting:     ref images → CLIP+MLP → style text →
           concatenate with prompt → Qwen-VL → Qwen-Image
           (Text-based aesthetic injection — the current pipeline default.)
  Mode C — Soft Prompting:     ref images → CLIP+MLP → PreferenceProjector →
           K virtual tokens injected into inputs_embeds → Qwen-VL → Qwen-Image
           (Our core contribution: continuous preference token injection.)

Usage:
  # Define test cases in the script or a JSON file, then:
  python run_experiments.py                          # run all test cases
  python run_experiments.py --cases 0                # run only case 0
  python run_experiments.py --mode C                 # run only Mode C
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image

# Allow sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (load_preference_encoder, extract_user_aesthetic_vector,
                       match_aesthetic_style, PersonalizationPipeline,
                       DEVICE, CHECKPOINT_PATH)
from qwen_bridge import PreferenceProjector, prepare_qwen_inputs_embeds

# ============================================================
# Config
# ============================================================

QWEN_VL_PATH = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct"
QWEN_HIDDEN_SIZE = 2560         # Qwen3-VL-4B-Instruct
NUM_VIRTUAL_TOKENS = 8          # K soft prompt tokens
PROJECTOR_HIDDEN = 1024

OUTPUT_DIR = Path(__file__).resolve().parent / "experiment_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Test Cases
# ============================================================

@dataclass
class TestCase:
    id: str
    short_prompt: str                           # user's short request
    ref_images: List[str] = field(default_factory=list)  # paths to reference images
    description: str = ""


# ---- Define your test cases here ----
TEST_CASES: List[TestCase] = [
    TestCase(
        id="cat_moonlight",
        short_prompt="一只小猫坐在月光下的窗台上",
        ref_images=[],   # ← fill with actual reference image paths
        description="Cat on a moonlit windowsill",
    ),
    TestCase(
        id="summer_poster",
        short_prompt="夏日海滩音乐节海报",
        ref_images=[],
        description="Summer beach music festival poster",
    ),
    TestCase(
        id="city_night",
        short_prompt="赛博朋克风格的未来城市夜景",
        ref_images=[],
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

    print(f"  [Qwen-VL] Loading {model_path}...")
    model_kwargs = {"dtype": "auto", "trust_remote_code": True}
    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs).eval()
    if "device_map" not in model_kwargs:
        model = model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def _unload_model(model, name: str = "model"):
    """Free GPU memory."""
    if model is not None:
        del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"  [Unload] {name} removed from GPU.")


def _generate_image_api(prompt: str, output_path: str,
                        size: str = "2048*2048",
                        negative_prompt: str = "") -> dict:
    """Call Qwen-Image API and save to output_path."""
    from pipeline import PersonalizationPipeline
    # We use a lightweight instance just for the API call
    pipe = PersonalizationPipeline(load_qwen_vl=False)
    return pipe.call_qwen_image_api(
        generated_prompt=prompt,
        output_dir=str(Path(output_path).parent),
        size=size,
        negative_prompt=negative_prompt,
        prompt_extend=False,  # we already enriched the prompt
    )


# ============================================================
# Mode A: Zero-shot Baseline
# ============================================================

def run_mode_a(short_prompt: str, output_path: str,
               max_new_tokens: int = 300,
               size: str = "2048*2048",
               negative_prompt: str = "") -> dict:
    """
    Mode A — Zero-shot Baseline.
    Direct Qwen-VL prompt enrichment. No preference module at all.
    Qwen-VL rewrites the short_prompt into a detailed T2I prompt freely.
    """
    print(f"\n{'#'*60}")
    print(f"# MODE A — Zero-shot Baseline")
    print(f"# Input: \"{short_prompt}\"")
    print(f"{'#'*60}\n")

    model, processor = _load_qwen_vl(QWEN_VL_PATH)

    # System instruction: prompt engineer, no aesthetic guidance
    system_instruction = (
        "You are an expert Text-to-Image prompt engineer. "
        "Rewrite the user's short request into ONE detailed, professional English prompt "
        "for an image generation model. "
        "Add concrete details about subject, composition, lighting, color palette, style, mood, "
        "camera or rendering quality, and artistic finish while preserving the user's intent. "
        "Output ONLY the English prompt, no extra text."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [{"type": "text", "text": short_prompt}]},
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

    print(f"  [Mode A] Enriched ({len(enriched.split())} words): {enriched[:200]}...")
    _unload_model(model, "Qwen3-VL")

    # ---- Generate image ----
    print(f"  [Mode A] Calling Qwen-Image API...")
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
# Mode B: Hard Prompting (Text-based Style Injection)
# ============================================================

def run_mode_b(short_prompt: str, ref_image_paths: List[str],
               output_path: str,
               max_new_tokens: int = 300,
               size: str = "2048*2048",
               negative_prompt: str = "") -> dict:
    """
    Mode B — Hard Prompting.
    Extract aesthetic preferences from reference images via CLIP+MLP,
    match top-K style descriptors as TEXT, concatenate into system prompt.
    This is the current pipeline default (text-based style injection).
    """
    print(f"\n{'#'*60}")
    print(f"# MODE B — Hard Prompting (Text-based Style)")
    print(f"# Input: \"{short_prompt}\", {len(ref_image_paths)} ref images")
    print(f"{'#'*60}\n")

    # 1. Load preference encoder
    print("  [Mode B] Loading preference encoder...")
    encoder = load_preference_encoder(str(CHECKPOINT_PATH), DEVICE)

    # 2. Extract preference vector & match styles
    pref_vector = extract_user_aesthetic_vector(ref_image_paths, encoder, DEVICE)
    style_text = match_aesthetic_style(pref_vector, encoder, top_k=5)

    # 3. Free encoder before loading Qwen-VL (VRAM)
    del encoder
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Load Qwen-VL with style-injected system prompt
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

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [{"type": "text", "text": short_prompt}]},
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

    print(f"  [Mode B] Enriched ({len(enriched.split())} words): {enriched[:200]}...")
    print(f"  [Mode B] Style text: {style_text[:150]}...")
    _unload_model(model, "Qwen3-VL")

    # 5. Generate image
    print(f"  [Mode B] Calling Qwen-Image API...")
    result = _generate_image_api(enriched, output_path, size=size,
                                 negative_prompt=negative_prompt)
    return {
        "mode": "B (Hard Prompting)",
        "enriched_prompt": enriched,
        "prompt_words": len(enriched.split()),
        "style_text": style_text,
        "output_path": output_path,
        "generation": result,
    }


# ============================================================
# Mode C: Soft Prompting (Virtual Token Injection)
# ============================================================

def run_mode_c(short_prompt: str, ref_image_paths: List[str],
               output_path: str,
               max_new_tokens: int = 300,
               size: str = "2048*2048",
               negative_prompt: str = "",
               num_virtual_tokens: int = NUM_VIRTUAL_TOKENS) -> dict:
    """
    Mode C — Soft Prompting (Our Core Contribution).

    Pipeline:
      1. ref images → CLIP ViT-L/14 → Image_MLP → 768-dim preference vector
      2. preference vector → PreferenceProjector → K virtual tokens [K, hidden_size]
      3. short_prompt tokenized → text embeddings
      4. [virtual_tokens | text_embeddings] → Qwen-VL → enriched prompt
      5. enriched prompt → Qwen-Image API → image

    Key difference from Mode B:
      - Mode B converts preference → text style descriptions → system prompt text
      - Mode C injects preference directly as continuous tokens (no text bottleneck)

    Note: The PreferenceProjector uses small random init (gain=0.1, zero bias).
    For production use, it should be trained end-to-end. In this experiment, it
    serves as a demonstration of the architecture.
    """
    print(f"\n{'#'*60}")
    print(f"# MODE C — Soft Prompting (Virtual Token Injection)")
    print(f"# Input: \"{short_prompt}\", {len(ref_image_paths)} ref images")
    print(f"# K={num_virtual_tokens} virtual tokens, hidden_size={QWEN_HIDDEN_SIZE}")
    print(f"{'#'*60}\n")

    if not ref_image_paths:
        print("  [Mode C] WARNING: No reference images. Using zero preference vector.")
        pref_vector = F.normalize(torch.zeros(768), dim=-1).to(DEVICE)
    else:
        # 1. Extract preference vector from reference images
        print("  [Mode C] Extracting preference vector from reference images...")
        encoder = load_preference_encoder(str(CHECKPOINT_PATH), DEVICE)
        pref_vector = extract_user_aesthetic_vector(ref_image_paths, encoder, DEVICE)
        # Also get style text for logging
        style_text = match_aesthetic_style(pref_vector, encoder, top_k=5)
        print(f"  [Mode C] Matched style: {style_text[:150]}...")
        del encoder
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Create PreferenceProjector and project → virtual tokens
    print(f"  [Mode C] Creating PreferenceProjector (768 → {num_virtual_tokens}×{QWEN_HIDDEN_SIZE})...")
    projector = PreferenceProjector(
        input_dim=768,
        num_virtual_tokens=num_virtual_tokens,
        qwen_hidden_size=QWEN_HIDDEN_SIZE,
        projector_hidden=PROJECTOR_HIDDEN,
    ).to(DEVICE)

    virtual_tokens = projector(pref_vector)  # [1, K, hidden_size]
    print(f"  [Mode C] Virtual tokens shape: {virtual_tokens.shape}")

    # 3. Load Qwen-VL
    model, processor = _load_qwen_vl(QWEN_VL_PATH)

    # 4. Build system instruction (generic, style info is in virtual tokens)
    system_instruction = (
        "You are an expert Text-to-Image prompt engineer. "
        "Write ONE detailed, professional English prompt for an image generation model. "
        "Include specifics about composition, lighting, color palette, style, mood, and artistic quality. "
        "Output ONLY the English prompt, no extra text."
    )

    # 5. Prepare inputs_embeds with virtual tokens
    # We need the tokenizer (not the full processor) for prepare_qwen_inputs_embeds
    tokenizer = processor.tokenizer

    inputs_embeds, attention_mask = prepare_qwen_inputs_embeds(
        model=model,
        tokenizer=tokenizer,
        virtual_tokens=virtual_tokens,
        text_prompt=short_prompt,
        system_prompt=system_instruction,
    )
    # Move to correct device and dtype
    inputs_embeds = inputs_embeds.to(device=model.device, dtype=model.dtype)
    attention_mask = attention_mask.to(device=model.device)
    print(f"  [Mode C] inputs_embeds: {inputs_embeds.shape}, "
          f"attention_mask: {attention_mask.shape}")

    # 6. Generate with inputs_embeds
    print(f"  [Mode C] Generating with soft prompt injection...")
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        except TypeError:
            # Fallback: some models require input_ids even with inputs_embeds
            # Create dummy input_ids from the text portion for compatibility
            print(f"  [Mode C] Model doesn't support inputs_embeds directly. "
                  f"Falling back to input_ids path...")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                {"role": "user", "content": [{"type": "text", "text": short_prompt}]},
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            # Generate with virtual tokens influencing the first forward pass
            generated_ids = model.generate(
                **inputs,
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

    # 7. Decode
    # The generated IDs include the full sequence; extract only new tokens
    input_len = attention_mask.size(1)  # K virtual tokens + text tokens
    enriched = tokenizer.decode(
        generated_ids[0][input_len:], skip_special_tokens=True
    ).strip()

    if not enriched:
        # Fallback decode: try decoding everything after the text portion
        text_only_len = attention_mask.size(1) - num_virtual_tokens
        enriched = tokenizer.decode(
            generated_ids[0][text_only_len:], skip_special_tokens=True
        ).strip()

    print(f"  [Mode C] Enriched ({len(enriched.split())} words): {enriched[:200]}...")

    _unload_model(model, "Qwen3-VL")
    del projector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 8. Generate image
    print(f"  [Mode C] Calling Qwen-Image API...")
    result = _generate_image_api(enriched, output_path, size=size,
                                 negative_prompt=negative_prompt)
    return {
        "mode": "C (Soft Prompting / Virtual Tokens)",
        "enriched_prompt": enriched,
        "prompt_words": len(enriched.split()),
        "num_virtual_tokens": num_virtual_tokens,
        "output_path": output_path,
        "generation": result,
    }


# ============================================================
# Experiment Runner
# ============================================================

def run_experiments(test_cases: List[TestCase],
                    modes: List[str] = None,
                    case_indices: List[int] = None) -> Dict:
    """
    Run all test cases through all specified modes.

    Args:
        test_cases: list of TestCase objects
        modes: which modes to run, e.g. ["A", "B", "C"]. None = all.
        case_indices: which cases to run, e.g. [0, 1]. None = all.

    Returns:
        dict with full results per case per mode.
    """
    if modes is None:
        modes = ["A", "B", "C"]

    if case_indices is not None:
        test_cases = [test_cases[i] for i in case_indices if i < len(test_cases)]

    all_results = {}

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"  TEST CASE: {case.id} — \"{case.short_prompt}\"")
        print(f"  Ref Images: {len(case.ref_images)} — {case.description}")
        print(f"{'='*70}")

        case_dir = OUTPUT_DIR / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_results = {"id": case.id, "short_prompt": case.short_prompt,
                        "num_ref_images": len(case.ref_images)}

        if "A" in modes:
            out_path = str(case_dir / f"{case.id}_mode_A.png")
            try:
                case_results["mode_A"] = run_mode_a(
                    case.short_prompt, out_path)
            except Exception as e:
                print(f"  [ERROR] Mode A failed: {e}")
                import traceback
                traceback.print_exc()
                case_results["mode_A"] = {"error": str(e)}

        if "B" in modes:
            out_path = str(case_dir / f"{case.id}_mode_B.png")
            try:
                case_results["mode_B"] = run_mode_b(
                    case.short_prompt, case.ref_images, out_path)
            except Exception as e:
                print(f"  [ERROR] Mode B failed: {e}")
                import traceback
                traceback.print_exc()
                case_results["mode_B"] = {"error": str(e)}

        if "C" in modes:
            out_path = str(case_dir / f"{case.id}_mode_C.png")
            try:
                case_results["mode_C"] = run_mode_c(
                    case.short_prompt, case.ref_images, out_path)
            except Exception as e:
                print(f"  [ERROR] Mode C failed: {e}")
                import traceback
                traceback.print_exc()
                case_results["mode_C"] = {"error": str(e)}

        all_results[case.id] = case_results

        # Save intermediate results
        report_path = OUTPUT_DIR / "experiment_results.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENTS COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Report: {OUTPUT_DIR / 'experiment_results.json'}")
    print(f"{'='*70}")

    return all_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-mode prompt strategy comparison experiment")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case indices, e.g. '0,1'. Default: all.")
    parser.add_argument("--mode", type=str, default=None,
                        help="Which mode(s): A, B, C, or comma-separated. Default: all.")
    parser.add_argument("--test-cases-json", type=str, default=None,
                        help="Path to JSON file with test cases.")
    parser.add_argument("--size", type=str, default="2048*2048",
                        help="Output image size.")
    args = parser.parse_args()

    # Load test cases
    test_cases = load_test_cases(args.test_cases_json)

    if not test_cases:
        print("ERROR: No test cases defined. Edit TEST_CASES in run_experiments.py "
              "or provide --test-cases-json.")
        return

    # Parse mode filter
    modes = None
    if args.mode:
        modes = [m.strip() for m in args.mode.split(",")]

    # Parse case filter
    case_indices = None
    if args.cases:
        case_indices = [int(i.strip()) for i in args.cases.split(",")]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT CONFIGURATION")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Modes: {modes or 'A, B, C'}")
    print(f"  Case filter: {case_indices or 'all'}")
    print(f"  Qwen-VL: {QWEN_VL_PATH}")
    print(f"  Preference checkpoint: {CHECKPOINT_PATH}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Image size: {args.size}")
    print(f"{'='*70}")

    results = run_experiments(
        test_cases=test_cases,
        modes=modes,
        case_indices=case_indices,
    )

    # Quick summary
    print(f"\n{'='*70}")
    print(f"  QUICK SUMMARY")
    print(f"{'='*70}")
    for case_id, case_data in results.items():
        print(f"\n  [{case_id}] \"{case_data['short_prompt']}\"")
        for mode_key in ["mode_A", "mode_B", "mode_C"]:
            if mode_key in case_data:
                m = case_data[mode_key]
                if "error" in m:
                    print(f"    {mode_key}: ❌ {m['error']}")
                else:
                    p = m.get("generation", {})
                    ok = "✅" if p.get("success") else "❌"
                    print(f"    {mode_key}: {ok}  {m['prompt_words']} words  "
                          f"→ {m.get('output_path', 'N/A')}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
