"""
Automated CLIP-based Evaluation for Mode A vs B vs C
=====================================================
Computes objective alignment metrics without needing FID (which requires
large reference distributions and InceptionV3).

Metrics:
  1. Text↔Image CLIP Score: cosine similarity between the user's original
     short_prompt and the generated image. Measures basic semantic fidelity
     — "did the image capture what the user asked for?"

  2. Image↔Image CLIP Score: cosine similarity between the user's reference
     images (mean embedding) and the generated image. Measures aesthetic /
     visual preference consistency — "does the output look like the user's
     reference style?"

For each test case, both scores are computed for Mode A, B, C.
Summary tables show per-case and aggregated results with Δ from baseline.

Usage:
  python evaluate_metrics.py                           # scan experiment_output/
  python evaluate_metrics.py --dir ./experiment_output # custom dir
  python evaluate_metrics.py --dir ./experiment_output --ref-dir ./my_refs
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Allow sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ============================================================
# Config
# ============================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CLIP Evaluator
# ============================================================

class CLIPEvaluator:
    """CLIP-based automated evaluation for generated images."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE):
        print(f"[CLIP Eval] Loading {model_name} on {device}...")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"  Ready.")

    # ----------------------------------------------------------
    # Metric 1: Text ↔ Image CLIP Score
    # ----------------------------------------------------------

    @torch.no_grad()
    def text_image_clip_score(self, prompt: str,
                               image_path: str) -> float:
        """
        Cosine similarity between the user's original short_prompt and the
        generated image. Higher = the image better captures the user's intent.

        Args:
            prompt:     the user's original short text request (NOT the enriched prompt)
            image_path: path to the generated image

        Returns:
            float in [-1, 1]. >0.25 = decent, >0.30 = good alignment.
        """
        # Encode text
        text_inputs = self.processor(
            text=[prompt], return_tensors="pt",
            padding=True, truncation=True, max_length=77
        ).to(self.device)
        text_emb = self.model.get_text_features(**text_inputs)
        text_emb = F.normalize(text_emb, dim=-1)  # [1, 512]

        # Encode image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open {image_path}: {e}")
            return None

        img_inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        img_emb = self.model.get_image_features(**img_inputs)
        img_emb = F.normalize(img_emb, dim=-1)  # [1, 512]

        score = (text_emb @ img_emb.T).item()
        return score

    # ----------------------------------------------------------
    # Metric 2: Image ↔ Image CLIP Score
    # ----------------------------------------------------------

    @torch.no_grad()
    def image_image_clip_score(self, ref_image_paths: List[str],
                                generated_image_path: str) -> Optional[float]:
        """
        Cosine similarity between the mean embedding of user reference images
        and the generated image. Measures visual style / aesthetic consistency.

        Args:
            ref_image_paths:       list of paths to user reference images
            generated_image_path:  path to the generated image

        Returns:
            float in [-1, 1]. Higher = output visually more similar to references.
            Returns None if no valid reference images.
        """
        if not ref_image_paths:
            return None

        # Encode reference images → mean embedding
        ref_embs = []
        for rp in ref_image_paths:
            try:
                img = Image.open(rp).convert("RGB")
            except Exception as e:
                print(f"  [WARN] Cannot open ref {rp}: {e}")
                continue
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = F.normalize(emb, dim=-1)
            ref_embs.append(emb)

        if not ref_embs:
            return None

        ref_mean = torch.stack(ref_embs).mean(dim=0)  # [512]
        ref_mean = F.normalize(ref_mean, dim=-1)

        # Encode generated image
        try:
            gen_img = Image.open(generated_image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open {generated_image_path}: {e}")
            return None

        gen_inputs = self.processor(images=gen_img, return_tensors="pt").to(self.device)
        gen_emb = self.model.get_image_features(**gen_inputs)
        gen_emb = F.normalize(gen_emb, dim=-1)  # [1, 512]

        score = (ref_mean @ gen_emb.T).item()
        return score

    # ----------------------------------------------------------
    # Batch Evaluation
    # ----------------------------------------------------------

    @torch.no_grad()
    def evaluate_case(self, case_id: str, short_prompt: str,
                       mode_images: Dict[str, str],
                       ref_image_paths: List[str] = None) -> Dict:
        """
        Evaluate one test case across all modes.

        Args:
            case_id:       test case identifier
            short_prompt:  user's original short text request
            mode_images:   dict like {"A": "path/to/mode_A.png", ...}
            ref_image_paths: paths to reference images (for Image↔Image score)

        Returns:
            dict with scores per mode.
        """
        ref_image_paths = ref_image_paths or []
        results = {"case_id": case_id, "short_prompt": short_prompt,
                   "ref_images": ref_image_paths, "scores": {}}

        for mode_name, img_path in sorted(mode_images.items()):
            if not img_path or not os.path.exists(img_path):
                print(f"  [{case_id}/{mode_name}] Image not found: {img_path}")
                results["scores"][mode_name] = {
                    "text_image_score": None,
                    "image_image_score": None,
                    "image_path": img_path,
                }
                continue

            ti_score = self.text_image_clip_score(short_prompt, img_path)
            ii_score = self.image_image_clip_score(ref_image_paths, img_path)

            ti_str = f"{ti_score:.4f}" if ti_score is not None else "N/A"
            ii_str = f"{ii_score:.4f}" if ii_score is not None else "N/A"
            print(f"  [{case_id}/{mode_name}] "
                  f"Text↔Image={ti_str}  "
                  f"Image↔Image={ii_str}")

            results["scores"][mode_name] = {
                "text_image_score": round(ti_score, 4) if ti_score is not None else None,
                "image_image_score": round(ii_score, 4) if ii_score is not None else None,
                "image_path": img_path,
            }

        return results


# ============================================================
# Auto-discovery: scan experiment output directory
# ============================================================

def discover_experiments(output_dir: str) -> List[Dict]:
    """
    Auto-discover experiments from directory structure.

    Expected layout (from run_experiments.py):
      experiment_output/
        cat_moonlight/
          cat_moonlight_mode_A.png
          cat_moonlight_mode_B.png
          cat_moonlight_mode_C.png
        summer_poster/
          summer_poster_mode_A.png
          ...

    Also reads experiment_results.json for prompts if available.

    Returns:
      list of dicts: [{"case_id": ..., "short_prompt": ..., "modes": {...}}, ...]
    """
    root = Path(output_dir)
    if not root.exists():
        print(f"[Discover] Directory not found: {root}")
        return []

    cases = []

    # Try loading experiment_results.json for metadata
    results_json = root / "experiment_results.json"
    exp_data = {}
    if results_json.exists():
        with open(results_json, "r", encoding="utf-8") as f:
            exp_data = json.load(f)

    # Scan subdirectories for images
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        case_id = subdir.name
        modes = {}
        for png in sorted(subdir.glob("*.png")):
            # Parse mode from filename: case_id_mode_X.png
            name = png.stem  # e.g. "cat_moonlight_mode_A"
            if "_mode_" in name:
                mode_letter = name.split("_mode_")[-1]  # "A", "B", "C"
                modes[mode_letter] = str(png)

        if not modes:
            continue

        # Get short_prompt from experiment_results.json
        short_prompt = ""
        if case_id in exp_data:
            short_prompt = exp_data[case_id].get("short_prompt", "")

        cases.append({
            "case_id": case_id,
            "short_prompt": short_prompt,
            "modes": modes,
        })

    print(f"[Discover] Found {len(cases)} test cases in {root}")
    for c in cases:
        print(f"  {c['case_id']}: modes {list(c['modes'].keys())} "
              f"prompt=\"{c['short_prompt'][:50]}\"")
    return cases


# ============================================================
# Report Generation
# ============================================================

def print_report(all_results: List[Dict]):
    """Print a formatted comparison table."""
    evaluator_name = "CLIP ViT-B/32"

    # ---- Aggregate across cases ----
    mode_scores = {}  # mode → {"ti": [scores], "ii": [scores]}
    for case in all_results:
        for mode_name, scores in case.get("scores", {}).items():
            if mode_name not in mode_scores:
                mode_scores[mode_name] = {"ti": [], "ii": []}
            ti = scores.get("text_image_score")
            ii = scores.get("image_image_score")
            if ti is not None:
                mode_scores[mode_name]["ti"].append(ti)
            if ii is not None:
                mode_scores[mode_name]["ii"].append(ii)

    print(f"\n{'='*80}")
    print(f"  AUTOMATED EVALUATION REPORT")
    print(f"  Evaluator: {evaluator_name} on {DEVICE}")
    print(f"  Test Cases: {len(all_results)}")
    print(f"{'='*80}")

    # ---- Per-Case Table ----
    print(f"\n{'─'*80}")
    print(f"  PER-CASE RESULTS")
    print(f"{'─'*80}")

    for case in all_results:
        print(f"\n  ▸ [{case['case_id']}] \"{case['short_prompt'][:60]}\"")
        print(f"    {'Mode':<40s} {'Text↔Image':>12s} {'Image↔Image':>14s}")
        print(f"    {'─'*40} {'─'*12} {'─'*14}")

        for mode_name in sorted(case.get("scores", {}).keys()):
            s = case["scores"][mode_name]
            ti = f"{s['text_image_score']:.4f}" if s['text_image_score'] is not None else "N/A"
            ii = f"{s['image_image_score']:.4f}" if s['image_image_score'] is not None else "N/A"

            mode_label = {
                "A": "A: Zero-shot Baseline",
                "B": "B: Hard Prompting (Hardcoded Text)",
                "C": "C: Ours (Preference Align)",
            }.get(mode_name, f"Mode {mode_name}")

            print(f"    {mode_label:<40s} {ti:>12s} {ii:>14s}")

    # ---- Aggregate Table ----
    print(f"\n{'─'*80}")
    print(f"  AGGREGATE RESULTS (mean ± std across {len(all_results)} cases)")
    print(f"{'─'*80}")
    print(f"  {'Mode':<40s} {'Text↔Image':>20s} {'Image↔Image':>20s}")
    print(f"  {'─'*40} {'─'*20} {'─'*20}")

    for mode_name in ["A", "B", "C"]:
        if mode_name not in mode_scores:
            continue
        ti_scores = mode_scores[mode_name]["ti"]
        ii_scores = mode_scores[mode_name]["ii"]

        ti_mean = torch.tensor(ti_scores).mean().item() if ti_scores else 0
        ti_std = torch.tensor(ti_scores).std().item() if len(ti_scores) > 1 else 0
        ii_mean = torch.tensor(ii_scores).mean().item() if ii_scores else 0
        ii_std = torch.tensor(ii_scores).std().item() if len(ii_scores) > 1 else 0

        mode_label = {
            "A": "A: Zero-shot Baseline",
            "B": "B: Hard Prompting (Hardcoded Text)",
            "C": "C: Ours (Preference Align)",
        }.get(mode_name, f"Mode {mode_name}")

        ti_str = f"{ti_mean:.4f} ± {ti_std:.4f}"
        ii_str = f"{ii_mean:.4f} ± {ii_std:.4f}" if ii_scores else "N/A"
        print(f"  {mode_label:<40s} {ti_str:>20s} {ii_str:>20s}")

    # ---- Delta from Baseline (Mode A) ----
    print(f"\n{'─'*80}")
    print(f"  DELTA FROM BASELINE (Mode A = Zero-shot)")
    print(f"{'─'*80}")

    if "A" in mode_scores:
        baseline_ti = torch.tensor(mode_scores["A"]["ti"]).mean().item()
        baseline_ii = torch.tensor(mode_scores["A"]["ii"]).mean().item() if mode_scores["A"]["ii"] else None

        print(f"  {'Mode':<40s} {'Δ Text↔Image':>16s} {'Δ Image↔Image':>16s} {'Winner':>10s}")
        print(f"  {'─'*40} {'─'*16} {'─'*16} {'─'*10}")

        for mode_name in ["B", "C"]:
            if mode_name not in mode_scores:
                continue
            ti_mean = torch.tensor(mode_scores[mode_name]["ti"]).mean().item()
            delta_ti = ti_mean - baseline_ti
            delta_ti_str = f"{delta_ti:+.4f}"

            delta_ii_str = "N/A"
            if baseline_ii is not None and mode_scores[mode_name]["ii"]:
                ii_mean = torch.tensor(mode_scores[mode_name]["ii"]).mean().item()
                delta_ii = ii_mean - baseline_ii
                delta_ii_str = f"{delta_ii:+.4f}"

            winner = "✓ BETTER" if delta_ti > 0 else ("✗ WORSE" if delta_ti < 0 else "= SAME")

            mode_label = {
                "B": "B: Hard Prompting vs A",
                "C": "C: Ours (Preference Align) vs A",
            }.get(mode_name, f"Mode {mode_name} vs A")

            print(f"  {mode_label:<40s} {delta_ti_str:>16s} {delta_ii_str:>16s} {winner:>10s}")

    # ---- Mode B vs C head-to-head ----
    if "B" in mode_scores and "C" in mode_scores:
        print(f"\n{'─'*80}")
        print(f"  HEAD-TO-HEAD: Mode C (Ours) vs Mode B (Hardcoded)")
        print(f"{'─'*80}")
        b_ti = torch.tensor(mode_scores["B"]["ti"]).mean().item()
        c_ti = torch.tensor(mode_scores["C"]["ti"]).mean().item()
        delta = c_ti - b_ti
        print(f"  Δ Text↔Image (C - B): {delta:+.4f}  "
              f"— {'C wins ✓' if delta > 0 else 'B wins ✓' if delta < 0 else 'Tie'}")
        if mode_scores["B"]["ii"] and mode_scores["C"]["ii"]:
            b_ii = torch.tensor(mode_scores["B"]["ii"]).mean().item()
            c_ii = torch.tensor(mode_scores["C"]["ii"]).mean().item()
            delta_ii = c_ii - b_ii
            print(f"  Δ Image↔Image (C - B): {delta_ii:+.4f}  "
                  f"— {'C wins ✓' if delta_ii > 0 else 'B wins ✓' if delta_ii < 0 else 'Tie'}")

    print(f"\n{'='*80}\n")


def save_report(all_results: List[Dict], output_path: str):
    """Save full evaluation results as JSON."""
    report = {
        "evaluator": CLIP_MODEL_NAME,
        "device": DEVICE,
        "num_cases": len(all_results),
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Full report saved to: {output_path}")


# ============================================================
# ImageGenerationEvaluator — used by webui.py
# ============================================================

class ImageGenerationEvaluator(CLIPEvaluator):
    """
    Extended evaluator for the Web UI providing clip_score(),
    aesthetic_score(), prompt_richness(), and evaluate() methods
    compatible with the webui.py interface.
    """

    def clip_score(self, prompt: str, image, ref_image=None) -> float:
        """
        Compute CLIP cosine similarity between prompt and generated image.

        Args:
            prompt:   the enriched T2I prompt
            image:    PIL Image of the generated image
            ref_image: (unused, kept for compatibility)
        Returns:
            float in [-1, 1]
        """
        if image is None:
            return 0.0
        import tempfile
        import os
        # Save PIL Image to temp file (CLIPEvaluator works with paths)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if hasattr(image, 'save'):
                image.save(tmp.name)
            img_path = tmp.name
        try:
            score = self.text_image_clip_score(prompt, img_path)
            return score if score is not None else 0.0
        finally:
            try:
                os.unlink(img_path)
            except Exception:
                pass

    def aesthetic_score(self, image) -> float:
        """
        Compute a heuristic aesthetic quality score for the image.

        Uses CLIP similarity against a set of high-quality aesthetic
        reference descriptors as a proxy for aesthetic quality.
        Higher = more aesthetically pleasing.
        """
        if image is None:
            return 0.0

        # Aesthetic reference descriptors (high-quality image traits)
        aesthetic_descriptors = [
            "a beautifully composed professional photograph, excellent lighting, sharp focus",
            "award-winning artistic composition, masterful use of color and contrast",
            "visually stunning high-quality image, rich detail and texture",
            "professionally shot photograph with perfect exposure and white balance",
        ]

        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if hasattr(image, 'save'):
                image.save(tmp.name)
            img_path = tmp.name
        try:
            # Encode image
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                return 0.0
            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            img_emb = self.model.get_image_features(**img_inputs)
            img_emb = F.normalize(img_emb, dim=-1)

            # Encode aesthetic descriptors
            scores = []
            for desc in aesthetic_descriptors:
                text_inputs = self.processor(
                    text=[desc], return_tensors="pt",
                    padding=True, truncation=True, max_length=77
                ).to(self.device)
                text_emb = self.model.get_text_features(**text_inputs)
                text_emb = F.normalize(text_emb, dim=-1)
                score = (text_emb @ img_emb.T).item()
                scores.append(score)

            return float(torch.tensor(scores).mean().item())
        finally:
            try:
                os.unlink(img_path)
            except Exception:
                pass

    def prompt_richness(self, prompt: str) -> dict:
        """
        Analyze prompt richness: word count, unique word ratio,
        and count of detail-oriented keywords.
        """
        if not prompt:
            return {"word_count": 0, "unique_ratio": 0.0, "detail_keywords": 0}

        import re
        words = re.findall(r'\b\w+\b', prompt.lower())
        word_count = len(words)
        unique_ratio = len(set(words)) / max(word_count, 1)

        # Count detail-oriented keywords
        detail_keywords = {
            "lighting", "shadow", "texture", "composition", "color",
            "detail", "resolution", "contrast", "palette", "atmosphere",
            "mood", "tone", "gradient", "focus", "bokeh", "depth",
            "reflection", "ambient", "volumetric", "cinematic",
            "photorealistic", "hyper", "professional", "elegant",
            "dramatic", "soft", "sharp", "warm", "cool", "golden",
            "artistic", "vibrant", "muted", "saturated", "desaturated",
            "layout", "typography", "negative space", "balance",
        }
        detail_count = sum(1 for w in words if w in detail_keywords)

        return {
            "word_count": word_count,
            "unique_ratio": round(unique_ratio, 4),
            "detail_keywords": detail_count,
        }

    def evaluate(self, prompt: str, generated_image,
                 reference_image=None) -> dict:
        """
        Full evaluation: CLIP score + aesthetic score + prompt richness.
        Compatible with webui.py's evaluate_image() callback.
        """
        cs = self.clip_score(prompt, generated_image)
        aes = self.aesthetic_score(generated_image)
        pr = self.prompt_richness(prompt)

        result = {
            "clip_score": round(cs, 4),
            "aesthetic_score": round(aes, 4),
            "prompt_richness": pr,
        }

        if reference_image is not None:
            result["style_consistency"] = round(
                self.clip_score(prompt, reference_image), 4)

        return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based automated evaluation for Mode A vs B vs C")
    parser.add_argument("--dir", type=str, default="./experiment_output",
                        help="Experiment output directory to scan")
    parser.add_argument("--ref-dir", type=str, default=None,
                        help="Directory containing reference images "
                             "(named <case_id>_ref_*.png)")
    parser.add_argument("--output", type=str, default="./evaluation_report.json",
                        help="Where to save detailed JSON report")
    args = parser.parse_args()

    # 1. Discover experiments
    cases = discover_experiments(args.dir)
    if not cases:
        print("ERROR: No experiments found. Run run_experiments.py first.")
        return

    # 2. Find reference images
    ref_mapping = {}  # case_id → list of ref image paths
    if args.ref_dir:
        ref_root = Path(args.ref_dir)
        for case in cases:
            cid = case["case_id"]
            refs = sorted(ref_root.glob(f"{cid}_ref_*"))
            if refs:
                ref_mapping[cid] = [str(r) for r in refs]
        print(f"[Refs] Found reference images for {len(ref_mapping)} cases.")

    # Also try experiment_results.json for ref image paths
    results_json = Path(args.dir) / "experiment_results.json"
    if results_json.exists():
        with open(results_json, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
        for cid, cdata in exp_data.items():
            if cid not in ref_mapping:
                refs = cdata.get("ref_images", [])
                if refs:
                    ref_mapping[cid] = refs

    # 3. Evaluate
    evaluator = CLIPEvaluator()
    all_results = []

    for case in cases:
        cid = case["case_id"]
        prompt = case["short_prompt"]
        refs = ref_mapping.get(cid, [])

        if not prompt:
            print(f"  [{cid}] WARNING: No short_prompt found. Using case_id as prompt.")
            prompt = cid

        result = evaluator.evaluate_case(
            case_id=cid,
            short_prompt=prompt,
            mode_images=case["modes"],
            ref_image_paths=refs,
        )
        all_results.append(result)

    # 4. Report
    print_report(all_results)
    save_report(all_results, args.output)


if __name__ == "__main__":
    main()
