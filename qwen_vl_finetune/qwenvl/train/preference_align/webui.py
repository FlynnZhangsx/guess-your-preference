"""
AI Image Studio — ChatGPT-style Interactive Web UI
--------------------------------------------------
Multi-turn preference-aligned image generation.

Usage:
    python webui.py                        # default port 7860
    python webui.py --port 8080            # custom port
"""

import os, sys, json, time, argparse, base64, io
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PersonalizationPipeline

# Patch gradio_client JSON schema bug
import gradio_client.utils as gc_utils
_o_get = gc_utils.get_type
_o_json = gc_utils._json_schema_to_python_type
def _p_get(s):
    if isinstance(s, bool): return "boolean"
    return _o_get(s)
def _p_json(s, d=None):
    if isinstance(s, bool): return "boolean"
    if not isinstance(s, dict): return str(type(s).__name__)
    return _o_json(s, d)
gc_utils.get_type = _p_get
gc_utils._json_schema_to_python_type = _p_json

QWEN_VL_PATH = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_pipeline = None

# ============================================================
# Helpers
# ============================================================

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _save_uploaded(uploaded) -> List[str]:
    """Save uploaded images (PIL from gr.Image or list) to temp dir."""
    paths = []
    tmp_dir = OUTPUT_DIR / "tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if uploaded is None:
        return paths

    items = uploaded if isinstance(uploaded, (list, tuple)) else [uploaded]
    for i, item in enumerate(items):
        try:
            if isinstance(item, Image.Image):
                p = tmp_dir / f"ref_{int(time.time())}_{i}.png"
                item.save(p)
                paths.append(str(p))
            elif isinstance(item, dict) and "image" in item:
                img = item["image"]
                if isinstance(img, Image.Image):
                    p = tmp_dir / f"ref_{int(time.time())}_{i}.png"
                    img.save(p)
                    paths.append(str(p))
            elif isinstance(item, str) and os.path.exists(item):
                paths.append(item)
        except Exception:
            pass
    return paths

def _render_chat(session: dict) -> str:
    """Render chat as clean HTML with user/AI bubbles."""
    history = session.get("turn_history", [])
    if not history:
        return '<div style="text-align:center;padding:60px 20px;color:#aaa;font-size:15px">Describe what you want to create and upload reference images</div>'

    bubbles = []
    for t in history:
        turn = t["turn"]
        feedback = t.get("feedback", "")
        prompt = t.get("prompt", "")
        img_path = t.get("image_path", "")
        cs = t.get("clip_score", 0)
        aes = t.get("aesthetic_score", 0)

        img_html = ""
        if img_path and os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_html = f'<img src="{_pil_to_b64(pil_img)}" style="max-width:100%;border-radius:12px;margin-top:8px">'
            except Exception:
                pass

        prompt_preview = prompt[:300] + ("..." if len(prompt) > 300 else "")

        bubbles.append(f'<div style="display:flex;justify-content:flex-end;margin:16px 0"><div style="max-width:72%;background:#f0f0f0;color:#222;padding:14px 18px;border-radius:16px 16px 4px 16px;font-size:14px;line-height:1.6"><div style="font-weight:600;font-size:12px;color:#888;margin-bottom:4px">You · Turn {turn}</div>{feedback}</div></div>')
        bubbles.append(f'<div style="display:flex;justify-content:flex-start;margin:16px 0"><div style="max-width:85%;background:#fff;color:#222;padding:14px 18px;border-radius:16px 16px 16px 4px;font-size:14px;line-height:1.6;border:1px solid #e5e5e5">{img_html}<div style="margin-top:8px;color:#555;font-size:13px;line-height:1.5">{prompt_preview}</div><div style="display:flex;gap:20px;margin-top:8px;font-size:12px"><span style="color:#667eea">CLIP {cs:.4f}</span><span style="color:#52c41a">Aesthetic {aes:.4f}</span></div></div></div>')

    return f'<div style="padding:8px">{"".join(bubbles)}</div>'

# ============================================================
# Callbacks (no progress — avoids SSE/proxy issues)
# ============================================================

def chat_generate(ref_img, refs_extra, prompt, mode, size, neg, tokens, sess):
    """Generate initial image. ref_img = single gr.Image, refs_extra = extra gr.Gallery."""
    global _pipeline
    if not prompt or not prompt.strip():
        return sess, _render_chat(sess), None, ""

    # Collect all reference images
    all_imgs = []
    if ref_img is not None:
        all_imgs.append(ref_img)
    if refs_extra is not None:
        items = refs_extra if isinstance(refs_extra, (list, tuple)) else [refs_extra]
        for item in items:
            if isinstance(item, (Image.Image, str)):
                all_imgs.append(item)
            elif isinstance(item, dict):
                all_imgs.append(item.get("image", item))

    paths = _save_uploaded(all_imgs if all_imgs else None)
    sess["ref_image_paths"] = paths

    try:
        _pipeline = PersonalizationPipeline(qwen_model_name=QWEN_VL_PATH, load_qwen_vl=True)
        enriched = _pipeline.generate_personalized_prompt(
            image_paths=paths, short_prompt=prompt.strip(), max_new_tokens=tokens,
        )
        _pipeline.unload_preference_encoder()
        _pipeline.unload_qwen_vl()

        result = _pipeline.generate_image_unified(
            prompt=enriched,
            mode="api" if mode == "API (DashScope)" else "local",
            size=size, negative_prompt=neg, output_dir=str(OUTPUT_DIR),
        )
    except Exception as e:
        return sess, _render_chat(sess), None, f"Error: {e}"

    if result.get("success"):
        img_path = result.get("image_path", "")
        gen_img = None
        try:
            gen_img = Image.open(img_path).convert("RGB")
        except Exception:
            pass

        cs = _clip_score(enriched, gen_img)
        aes = _aesthetic_score(gen_img)

        t = {"turn": len(sess.get("turn_history", [])) + 1, "feedback": prompt.strip(),
             "prompt": enriched, "image_path": img_path, "clip_score": cs, "aesthetic_score": aes}
        sess.setdefault("turn_history", []).append(t)
        sess["current_prompt"] = enriched
        sess["current_image_path"] = img_path
        return sess, _render_chat(sess), gen_img, f"CLIP {cs:.4f}  ·  Aesthetic {aes:.4f}"

    return sess, _render_chat(sess), None, f"Failed: {result.get('error')}"


def chat_refine(feedback, mode, size, neg, tokens, sess):
    """Refine current image with feedback."""
    global _pipeline
    if not sess.get("current_prompt"):
        return sess, _render_chat(sess), None, "Generate an image first."
    if not feedback or not feedback.strip():
        return sess, _render_chat(sess), None, ""

    try:
        _pipeline = PersonalizationPipeline(qwen_model_name=QWEN_VL_PATH, load_qwen_vl=True)
        if sess.get("ref_image_paths"):
            try: _pipeline._load_preference_encoder_if_needed()
            except Exception: pass

        refined = _pipeline.refine_prompt_with_feedback(
            current_prompt=sess["current_prompt"], user_feedback=feedback.strip(),
            previous_image_path=sess.get("current_image_path"),
            image_paths=sess.get("ref_image_paths"), max_new_tokens=tokens,
        )
        _pipeline.unload_preference_encoder()
        _pipeline.unload_qwen_vl()

        result = _pipeline.generate_image_unified(
            prompt=refined,
            mode="api" if mode == "API (DashScope)" else "local",
            size=size, negative_prompt=neg, output_dir=str(OUTPUT_DIR),
        )
    except Exception as e:
        return sess, _render_chat(sess), None, f"Error: {e}"

    if result.get("success"):
        img_path = result.get("image_path", "")
        gen_img = None
        try:
            gen_img = Image.open(img_path).convert("RGB")
        except Exception:
            pass

        cs = _clip_score(refined, gen_img)
        aes = _aesthetic_score(gen_img)

        t = {"turn": len(sess.get("turn_history", [])) + 1, "feedback": feedback.strip(),
             "prompt": refined, "image_path": img_path, "clip_score": cs, "aesthetic_score": aes}
        sess.setdefault("turn_history", []).append(t)
        sess["current_prompt"] = refined
        sess["current_image_path"] = img_path
        return sess, _render_chat(sess), gen_img, f"CLIP {cs:.4f}  ·  Aesthetic {aes:.4f}"

    return sess, _render_chat(sess), None, f"Failed: {result.get('error')}"


def chat_reset():
    empty = {"ref_image_paths": [], "turn_history": [], "current_prompt": "", "current_image_path": ""}
    return empty, _render_chat(empty), None, ""


# Simple in-process CLIP evaluator (no separate class needed)
_eval = None
def _get_eval():
    global _eval
    if _eval is None:
        from evaluate_metrics import ImageGenerationEvaluator
        _eval = ImageGenerationEvaluator()
    return _eval

def _clip_score(prompt, image):
    if not image or not prompt: return 0.0
    try: return _get_eval().clip_score(prompt, image)
    except Exception: return 0.0

def _aesthetic_score(image):
    if not image: return 0.0
    try: return _get_eval().aesthetic_score(image)
    except Exception: return 0.0

# ============================================================
# UI
# ============================================================

def build_ui():
    theme = gr.themes.Soft(primary_hue="gray", secondary_hue="gray", neutral_hue="gray").set(
        body_background_fill="#fafafa",
        block_background_fill="#ffffff",
        block_border_color="#e5e5e5",
        block_border_width="1px",
        block_radius="12px",
        input_background_fill="#ffffff",
        input_border_color="#d0d0d0",
        button_primary_background_fill="#1a1a1a",
        button_primary_background_fill_hover="#333333",
        button_primary_text_color="#ffffff",
    )

    css = """
    footer{display:none!important}
    .gradio-container{max-width:900px!important;margin:0 auto!important}
    #chat-panel{min-height:400px;max-height:60vh;overflow-y:auto}
    ::-webkit-scrollbar{width:5px}
    ::-webkit-scrollbar-track{background:transparent}
    ::-webkit-scrollbar-thumb{background:#ddd;border-radius:3px}
    """

    with gr.Blocks(theme=theme, css=css, title="AI Image Studio") as app:
        sess = gr.State({"ref_image_paths": [], "turn_history": [], "current_prompt": "", "current_image_path": ""})

        gr.HTML('<div style="text-align:center;padding:24px 0 8px"><h1 style="font-size:22px;font-weight:600;color:#1a1a1a;margin:0;letter-spacing:-0.5px">AI Image Studio</h1><p style="color:#999;margin:6px 0 0;font-size:13px">Reference Images + Text → Enriched Prompt → Image</p></div>')

        with gr.Tabs():
            # ============================================
            # TAB: Chat
            # ============================================
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("#### Reference Image")
                        ref_img = gr.Image(label=None, type="pil", show_label=False, height=160)

                        gr.Markdown("#### Settings")
                        mode_r = gr.Radio(["API (DashScope)", "Local (diffusers)"], value="API (DashScope)", label="Engine")
                        size_d = gr.Dropdown(["1024*1024", "1664*928", "2048*2048"], value="2048*2048", label="Size")
                        neg_t = gr.Textbox(label="Negative Prompt", value="low quality, blurry, distorted", lines=1, max_lines=2)
                        with gr.Accordion("Advanced", open=False):
                            tok_s = gr.Slider(100, 500, value=300, step=50, label="Max Tokens")
                        reset_btn = gr.Button("New Session", size="sm")

                    with gr.Column(scale=3):
                        chat_html = gr.HTML(value=_render_chat({"turn_history": []}), elem_id="chat-panel")

                        with gr.Group():
                            prompt_t = gr.Textbox(label=None, show_label=False, placeholder="Describe the image you want to create...", lines=1, scale=5)
                            with gr.Row():
                                gen_btn = gr.Button("Generate", variant="primary")
                                dummy_img = gr.Image(type="pil", visible=False)
                        with gr.Group():
                            refine_t = gr.Textbox(label=None, show_label=False, placeholder="How to change it? e.g. warmer, add flowers...", lines=1, scale=5)
                            with gr.Row():
                                refine_btn = gr.Button("Refine")
                                dummy_img2 = gr.Image(type="pil", visible=False)
                        status_t = gr.Textbox(label=None, show_label=False, interactive=False, container=False)

                gen_btn.click(fn=chat_generate, inputs=[ref_img, gr.State(None), prompt_t, mode_r, size_d, neg_t, tok_s, sess], outputs=[sess, chat_html, dummy_img, status_t]).then(fn=lambda: "", inputs=[], outputs=[prompt_t])
                refine_btn.click(fn=chat_refine, inputs=[refine_t, mode_r, size_d, neg_t, tok_s, sess], outputs=[sess, chat_html, dummy_img2, status_t]).then(fn=lambda: "", inputs=[], outputs=[refine_t])
                refine_t.submit(fn=chat_refine, inputs=[refine_t, mode_r, size_d, neg_t, tok_s, sess], outputs=[sess, chat_html, dummy_img2, status_t]).then(fn=lambda: "", inputs=[], outputs=[refine_t])
                reset_btn.click(fn=chat_reset, inputs=[], outputs=[sess, chat_html, dummy_img, status_t])

        gr.HTML('<div style="text-align:center;padding:20px;color:#ccc;font-size:11px">CLIP ViT-L/14 · Qwen3-VL · Qwen-Image</div>')

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app = build_ui()
    # NO queue — avoids WebSocket dependency through proxies
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
