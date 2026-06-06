"""
Personalized Prompt-to-Image Generation — Interactive Web UI
-------------------------------------------------------------
Gradio-based UI for the full preference-alignment pipeline.

Tabs:
  1. Generate — Single image generation with full control
  2. Compare — Side-by-side strategy comparison
  3. Evaluate — CLIP score + human ratings on generated images

Usage:
    python webui.py                        # default port 7860
    python webui.py --port 8080            # custom port
    python webui.py --share                # public link via Gradio
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

# Ensure we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import PersonalizationPipeline
from eval_metrics import ImageGenerationEvaluator

# ============================================================
# Config
# ============================================================

QWEN_VL_PATH = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RATINGS_DIR = Path(__file__).resolve().parent / "human_ratings"
RATINGS_DIR.mkdir(exist_ok=True)

# ============================================================
# Global pipeline & evaluator (lazy init per session)
# ============================================================

_pipeline: Optional[PersonalizationPipeline] = None
_evaluator: Optional[ImageGenerationEvaluator] = None


def get_evaluator() -> ImageGenerationEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = ImageGenerationEvaluator()
    return _evaluator


# ============================================================
# Tab 1: Single Generation
# ============================================================

def generate_single(
    ref_images: List[Image.Image],
    short_prompt: str,
    mode: str,
    size: str,
    negative_prompt: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
    progress=gr.Progress(),
) -> Tuple[str, str, Image.Image, str]:
    """
    Run the full pipeline: ref images + short prompt → enriched prompt → image.
    """
    global _pipeline

    if not short_prompt.strip():
        return "", "", None, "⚠️ 请输入提示词 (Please enter a prompt)."

    # --- Save reference images to temp files ---
    image_paths = []
    tmp_dir = OUTPUT_DIR / "tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(ref_images):
        if img is not None:
            p = tmp_dir / f"ref_{int(time.time())}_{i}.png"
            img.save(p)
            image_paths.append(str(p))

    progress(0.1, desc="Loading Qwen3-VL...")

    # --- Load pipeline ---
    try:
        _pipeline = PersonalizationPipeline(
            qwen_model_name=QWEN_VL_PATH,
            load_qwen_vl=True,
        )
    except Exception as e:
        return "", "", None, f"❌ Failed to load Qwen3-VL: {e}"

    progress(0.3, desc="Extracting preferences & generating prompt...")

    # --- Generate enriched prompt ---
    try:
        enriched_prompt = _pipeline.generate_personalized_prompt(
            image_paths=image_paths,
            short_prompt=short_prompt,
            max_new_tokens=max_tokens,
        )
    except Exception as e:
        return "", "", None, f"❌ Prompt generation failed: {e}"

    progress(0.6, desc="Unloading Qwen3-VL...")

    # --- Free memory ---
    _pipeline.unload_preference_encoder()
    _pipeline.unload_qwen_vl()

    progress(0.75, desc=f"Generating image via {mode}...")

    # --- Generate image ---
    try:
        result = _pipeline.generate_image_unified(
            prompt=enriched_prompt,
            mode="api" if mode == "API (DashScope)" else "local",
            size=size,
            negative_prompt=negative_prompt,
            output_dir=str(OUTPUT_DIR),
        )
    except Exception as e:
        return enriched_prompt, "", None, f"❌ Image generation failed: {e}"

    progress(0.95, desc="Done!")

    if result.get("success"):
        image_path = result.get("image_path", "")
        try:
            gen_image = Image.open(image_path).convert("RGB")
        except Exception:
            gen_image = None

        # Compute CLIP score
        ev = get_evaluator()
        cs = ev.clip_score(enriched_prompt, gen_image) if gen_image else 0.0
        aes = ev.aesthetic_score(gen_image) if gen_image else 0.0
        pr = ev.prompt_richness(enriched_prompt)

        status = (
            f"✅ **生成成功** | CLIP Score: `{cs:.4f}` | Aesthetic: `{aes:.4f}`\n\n"
            f"**Enriched Prompt** ({pr['word_count']} words, "
            f"unique ratio={pr['unique_ratio']}, "
            f"detail keywords={pr['detail_keywords']}):\n\n"
            f"{enriched_prompt}\n\n"
            f"**Image saved to:** `{image_path}`"
        )
        return enriched_prompt, image_path, gen_image, status
    else:
        return enriched_prompt, "", None, f"❌ Failed: {result.get('error', 'Unknown error')}"


# ============================================================
# Tab 2: Compare Strategies
# ============================================================

def compare_strategies(
    ref_images: List[Image.Image],
    short_prompt: str,
    size: str,
    negative_prompt: str,
    progress=gr.Progress(),
) -> Tuple[Image.Image, Image.Image, str, str, str]:
    """
    Run 0img (neutral) vs Nimg (preference-aligned) side-by-side.
    """
    global _pipeline

    if not short_prompt.strip():
        return None, None, "", "", "⚠️ 请输入提示词"

    # Save reference images
    image_paths = []
    tmp_dir = OUTPUT_DIR / "tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(ref_images):
        if img is not None:
            p = tmp_dir / f"comp_ref_{int(time.time())}_{i}.png"
            img.save(p)
            image_paths.append(str(p))

    progress(0.1, desc="Loading Qwen3-VL...")
    _pipeline = PersonalizationPipeline(qwen_model_name=QWEN_VL_PATH, load_qwen_vl=True)

    # ---- Strategy A: Neutral centroid (no ref images) ----
    progress(0.2, desc="Strategy A: Neutral centroid...")
    prompt_A = _pipeline.generate_personalized_prompt(
        image_paths=[],
        short_prompt=short_prompt,
    )
    _pipeline.unload_preference_encoder()

    prompt_B = prompt_A
    if image_paths:
        progress(0.5, desc="Strategy B: Preference-aligned...")
        prompt_B = _pipeline.generate_personalized_prompt(
            image_paths=image_paths,
            short_prompt=short_prompt,
        )

    _pipeline.unload_preference_encoder()
    _pipeline.unload_qwen_vl()

    # Generate images
    progress(0.7, desc="Generating image A (neutral)...")
    result_A = _pipeline.generate_image_unified(
        prompt=prompt_A, mode="api", size=size,
        negative_prompt=negative_prompt, output_dir=str(OUTPUT_DIR),
    )

    progress(0.85, desc="Generating image B (aligned)...")
    result_B = _pipeline.generate_image_unified(
        prompt=prompt_B, mode="api", size=size,
        negative_prompt=negative_prompt, output_dir=str(OUTPUT_DIR),
    )

    progress(0.95, desc="Evaluating...")

    # Load images and compute scores
    img_A = Image.open(result_A["image_path"]).convert("RGB") if result_A.get("success") else None
    img_B = Image.open(result_B["image_path"]).convert("RGB") if result_B.get("success") else None

    ev = get_evaluator()

    score_A = ev.clip_score(prompt_A, img_A) if img_A else 0
    aes_A = ev.aesthetic_score(img_A) if img_A else 0
    score_B = ev.clip_score(prompt_B, img_B) if img_B else 0
    aes_B = ev.aesthetic_score(img_B) if img_B else 0

    status_A = f"**Strategy A: 0-Image (Neutral Centroid)**\n\nCLIP: `{score_A:.4f}` | Aesthetic: `{aes_A:.4f}`\n\n{prompt_A[:500]}"
    status_B = f"**Strategy B: {len(image_paths)}-Image (Preference Aligned)**\n\nCLIP: `{score_B:.4f}` | Aesthetic: `{aes_B:.4f}`\n\n{prompt_B[:500]}"

    # Comparison summary
    diff_clip = score_B - score_A
    diff_aes = aes_B - aes_A
    winner = "B (Preference-Aligned)" if diff_clip > 0 else "A (Neutral Centroid)"
    summary = (
        f"### 📊 Comparison Summary\n\n"
        f"| Metric | A: Neutral | B: Aligned | Δ |\n"
        f"|--------|-----------|-----------|-----|\n"
        f"| CLIP Score | {score_A:.4f} | {score_B:.4f} | {diff_clip:+.4f} |\n"
        f"| Aesthetic | {aes_A:.4f} | {aes_B:.4f} | {diff_aes:+.4f} |\n\n"
        f"**Winner by CLIP Score: {winner}**"
    )

    return img_A, img_B, status_A, status_B, summary


# ============================================================
# Tab 3: Evaluation
# ============================================================

def evaluate_image(
    image: Image.Image,
    prompt: str,
    ref_image: Optional[Image.Image],
) -> Tuple[str, str]:
    """Compute CLIP metrics for a single image."""
    if image is None:
        return "⚠️ 请上传图片", ""
    if not prompt.strip():
        prompt = "A high-quality generated image"

    ev = get_evaluator()
    ref = ref_image if ref_image is not None else None
    results = ev.evaluate(prompt=prompt, generated_image=image, reference_image=ref)

    score_text = (
        f"### 📊 Auto Metrics\n\n"
        f"| Metric | Score |\n"
        f"|--------|-------|\n"
        f"| **CLIP Score** (prompt↔image) | `{results['clip_score']:.4f}` |\n"
        f"| **Aesthetic Score** (quality) | `{results['aesthetic_score']:.4f}` |\n"
    )
    if ref is not None:
        score_text += f"| **Style Consistency** (ref↔output) | `{results.get('style_consistency', 0):.4f}` |\n"

    score_text += (
        f"\n**Prompt Richness:** {results['prompt_richness']['word_count']} words, "
        f"{results['prompt_richness']['unique_ratio']} unique ratio, "
        f"{results['prompt_richness']['detail_keywords']} detail keywords."
    )

    return score_text, json.dumps(results, ensure_ascii=False, indent=2)


def submit_rating(
    image: Image.Image,
    prompt: str,
    aesthetics_rating: float,
    alignment_rating: float,
    notes: str,
) -> str:
    """Save human rating to JSON."""
    if image is None:
        return "⚠️ 请先生成或上传图片"

    rating = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt[:500] if prompt else "",
        "aesthetics_rating": int(aesthetics_rating),
        "alignment_rating": int(alignment_rating),
        "notes": notes,
    }

    # Save to ratings file
    ratings_file = RATINGS_DIR / f"ratings_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(ratings_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rating, ensure_ascii=False) + "\n")

    # Also save the rated image
    img_file = RATINGS_DIR / f"rated_{datetime.now().strftime('%H%M%S')}.png"
    image.save(img_file)

    # Load aggregate stats
    all_ratings = []
    for rf in sorted(RATINGS_DIR.glob("ratings_*.jsonl")):
        for line in rf.read_text().splitlines():
            try:
                all_ratings.append(json.loads(line))
            except Exception:
                pass

    n = len(all_ratings)
    if n > 0:
        avg_aes = sum(r["aesthetics_rating"] for r in all_ratings) / n
        avg_align = sum(r["alignment_rating"] for r in all_ratings) / n
        return (
            f"✅ **Rating submitted!** (Total: {n} ratings)\n\n"
            f"### 📈 Aggregate Statistics\n\n"
            f"| Metric | Average |\n"
            f"|--------|--------|\n"
            f"| Aesthetics (1-5) | `{avg_aes:.2f}` |\n"
            f"| Alignment (1-5) | `{avg_align:.2f}` |\n\n"
            f"Ratings saved to: `{ratings_file}`\n"
            f"Rated image: `{img_file}`"
        )
    return f"✅ Rating submitted!"


# ============================================================
# Build Gradio UI
# ============================================================

def build_ui():
    theme = gr.themes.Soft(primary_hue="blue")

    with gr.Blocks(theme=theme, title="Personalized Prompt-to-Image Generator") as app:
        gr.Markdown("""
        # 🎨 Personalized Prompt-to-Image Generator
        ### Preference Alignment Pipeline: Reference Images + Short Text → Enriched Prompt → Qwen-Image

        **Pipeline:** CLIP ViT-L/14 → Image_MLP → Style Matching → Qwen3-VL-4B → Qwen-Image API
        """)

        with gr.Tabs():
            # ================================================
            # TAB 1: Generate
            # ================================================
            with gr.TabItem("🎯 Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Inputs")
                        ref_imgs = gr.File(
                            label="Reference Images (0-N, optional)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        short_prompt = gr.Textbox(
                            label="Short Prompt",
                            placeholder="一只小猫坐在月光下的窗台上",
                            lines=2,
                        )
                        with gr.Row():
                            mode_radio = gr.Radio(
                                choices=["API (DashScope)", "Local (diffusers)"],
                                value="API (DashScope)",
                                label="Generation Mode",
                            )
                        size_dd = gr.Dropdown(
                            choices=["1024*1024", "1664*928", "2048*2048"],
                            value="2048*2048",
                            label="Output Size",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感",
                            lines=2,
                        )
                        with gr.Accordion("Advanced Options", open=False):
                            top_k = gr.Slider(1, 10, value=5, step=1,
                                              label="Top-K Style Descriptors")
                            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1,
                                                    label="Qwen-VL Temperature")
                            max_tokens = gr.Slider(100, 500, value=300, step=50,
                                                   label="Max Prompt Tokens")

                        gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Outputs")
                        enriched_prompt_out = gr.Textbox(
                            label="Enriched Prompt", lines=6, interactive=False)
                        image_path_out = gr.Textbox(
                            label="Image Path", interactive=False, visible=False)
                        gen_image_out = gr.Image(label="Generated Image", type="pil")
                        status_out = gr.Markdown("")

                gen_btn.click(
                    fn=generate_single,
                    inputs=[ref_imgs, short_prompt, mode_radio, size_dd,
                            neg_prompt, top_k, temperature, max_tokens],
                    outputs=[enriched_prompt_out, image_path_out, gen_image_out, status_out],
                )

            # ================================================
            # TAB 2: Compare
            # ================================================
            with gr.TabItem("🔬 Compare Strategies"):
                gr.Markdown("""
                ### Side-by-side comparison: 0-image (neutral) vs N-image (preference-aligned)
                Upload reference images to see how preference alignment changes the output.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        cmp_refs = gr.File(
                            label="Reference Images",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        cmp_prompt = gr.Textbox(
                            label="Short Prompt",
                            placeholder="一只小猫坐在月光下的窗台上",
                            lines=2,
                        )
                        cmp_size = gr.Dropdown(
                            choices=["1024*1024", "1664*928", "2048*2048"],
                            value="2048*2048", label="Output Size",
                        )
                        cmp_neg = gr.Textbox(
                            label="Negative Prompt",
                            value="低分辨率，低画质，肢体畸形",
                            lines=1,
                        )
                        cmp_btn = gr.Button("🔬 Run Comparison", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Row():
                            cmp_img_A = gr.Image(label="A: Neutral Centroid (0-img)", type="pil")
                            cmp_img_B = gr.Image(label="B: Preference Aligned (N-img)", type="pil")
                        with gr.Row():
                            cmp_status_A = gr.Markdown("", label="Strategy A")
                            cmp_status_B = gr.Markdown("", label="Strategy B")
                        cmp_summary = gr.Markdown("")

                cmp_btn.click(
                    fn=compare_strategies,
                    inputs=[cmp_refs, cmp_prompt, cmp_size, cmp_neg],
                    outputs=[cmp_img_A, cmp_img_B, cmp_status_A, cmp_status_B, cmp_summary],
                )

            # ================================================
            # TAB 3: Evaluate
            # ================================================
            with gr.TabItem("📊 Evaluate"):
                gr.Markdown("""
                ### Automated Metrics + Human Ratings
                Upload a generated image to compute CLIP Score and Aesthetic Score.
                Optionally rate the image for aesthetics and prompt alignment.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_image = gr.Image(label="Generated Image", type="pil")
                        eval_prompt = gr.Textbox(
                            label="T2I Prompt Used",
                            placeholder="Paste the enriched prompt here...",
                            lines=3,
                        )
                        eval_ref = gr.Image(label="Reference Image (optional)", type="pil")
                        eval_btn = gr.Button("📊 Compute Metrics", variant="primary")

                    with gr.Column(scale=1):
                        eval_scores = gr.Markdown("### Scores will appear here...")
                        eval_json = gr.Textbox(label="Raw JSON", visible=False)

                eval_btn.click(
                    fn=evaluate_image,
                    inputs=[eval_image, eval_prompt, eval_ref],
                    outputs=[eval_scores, eval_json],
                )

                gr.Markdown("---")
                gr.Markdown("### ⭐ Human Rating")

                with gr.Row():
                    with gr.Column(scale=1):
                        rate_image = gr.Image(label="Image to Rate", type="pil")
                        rate_prompt = gr.Textbox(
                            label="Prompt Used",
                            placeholder="Paste the enriched prompt...",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        aes_slider = gr.Slider(1, 5, value=3, step=1,
                                               label="Aesthetics (1=Poor, 5=Excellent)")
                        align_slider = gr.Slider(1, 5, value=3, step=1,
                                                 label="Prompt Alignment (1=Poor, 5=Perfect)")
                        rate_notes = gr.Textbox(label="Notes (optional)", lines=2)
                        rate_btn = gr.Button("⭐ Submit Rating", variant="secondary")
                        rate_status = gr.Markdown("")

                rate_btn.click(
                    fn=submit_rating,
                    inputs=[rate_image, rate_prompt, aes_slider, align_slider, rate_notes],
                    outputs=[rate_status],
                )

        # Footer
        gr.Markdown("""
        ---
        **Pipeline:** CLIP ViT-L/14 + Image_MLP → Top-K Style Matching → Qwen3-VL-4B-Instruct → Qwen-Image API

        **Models:** Preference Align (Val AUC=0.8934) | Qwen3-VL-4B-Instruct | qwen-image-2.0-pro
        """)

    return app


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Web UI for personalized image generation")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    app = build_ui()
    app.queue(max_size=10).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
