"""
AI Image Studio — FastAPI Backend
=================================
Serves the HTML frontend and REST API for image generation.

Usage:
    python api_server.py --port 7860
"""
import os, sys, json, time, uuid, argparse, base64, io, shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PersonalizationPipeline

# ============================================================
# Config
# ============================================================
QWEN_VL_PATH = "/home/coder/project/data/mllm/models/Qwen3-VL-4B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
for d in [OUTPUT_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global pipeline & session
_pipeline: Optional[PersonalizationPipeline] = None
_session: dict = {"ref_image_paths": [], "turn_history": [], "current_prompt": "", "current_image_path": ""}

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AI Image Studio", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def _get_pipeline() -> PersonalizationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PersonalizationPipeline(qwen_model_name=QWEN_VL_PATH, load_qwen_vl=True)
    return _pipeline

# Pre-load at startup
_models_ready = False

@app.on_event("startup")
async def startup():
    global _models_ready
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("Pre-loading models (this may take 1-2 minutes on first run)...")
    try:
        pipe = _get_pipeline()
        # Pre-load the preference encoder too
        pipe._load_preference_encoder_if_needed()
        # Generate a dummy prompt to warm up Qwen-VL
        pipe.generate_personalized_prompt(image_paths=[], short_prompt="test", max_new_tokens=10)
        _models_ready = True
        logger.info("All models loaded and ready!")
    except Exception as e:
        logger.error(f"Model pre-loading failed: {e}")
        _models_ready = False

# ============================================================
# Static files & routes
# ============================================================
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


def _img_path_to_url(path: str) -> str:
    """Convert local image path to relative URL."""
    if not path: return ""
    fname = Path(path).name
    return f"/output/{fname}"

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def api_status():
    return JSONResponse({"ready": _models_ready, "message": "Models ready" if _models_ready else "Loading models..."})


@app.post("/api/generate")
async def api_generate(
    prompt: str = Form(...),
    mode: str = Form("api"),
    size: str = Form("2048*2048"),
    negative_prompt: str = Form("low quality, blurry, distorted"),
    max_tokens: int = Form(300),
    ref_images: List[UploadFile] = File([]),
):
    """Generate an image from prompt + optional reference images."""
    global _session

    if not prompt.strip():
        raise HTTPException(400, "Prompt is required")

    # Save reference images
    image_paths = []
    for f in (ref_images or []):
        try:
            content = await f.read()
            if content:
                p = UPLOAD_DIR / f"ref_{uuid.uuid4().hex[:8]}_{f.filename or 'img.png'}"
                p.write_bytes(content)
                image_paths.append(str(p))
        except Exception:
            pass

    _session["ref_image_paths"] = image_paths

    try:
        pipe = _get_pipeline()
        enriched = pipe.generate_personalized_prompt(
            image_paths=image_paths, short_prompt=prompt.strip(), max_new_tokens=max_tokens,
        )

        result = pipe.generate_image_unified(
            prompt=enriched,
            mode="api" if "api" in mode.lower() else "local",
            size=size, negative_prompt=negative_prompt, output_dir=str(OUTPUT_DIR),
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    if not result.get("success"):
        _unload()
        raise HTTPException(500, result.get("error", "Unknown error"))

    img_path = result.get("image_path", "")
    img_url = _img_path_to_url(img_path)
    pil_img = None
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        pass

    # Score
    try:
        from evaluate_metrics import ImageGenerationEvaluator
        ev = ImageGenerationEvaluator()
        cs = ev.clip_score(enriched, pil_img) if pil_img else 0
        aes = ev.aesthetic_score(pil_img) if pil_img else 0
    except Exception:
        cs = aes = 0

    _session["turn_history"] = [{
        "turn": 1, "feedback": prompt.strip(), "prompt": enriched,
        "image_url": img_url, "clip_score": round(cs, 4), "aesthetic_score": round(aes, 4),
    }]
    _session["current_prompt"] = enriched
    _session["current_image_path"] = img_path

    return JSONResponse({
        "success": True,
        "image_url": img_url,
        "prompt": enriched,
        "clip_score": round(cs, 4),
        "aesthetic_score": round(aes, 4),
        "turn": 1,
    })


@app.post("/api/refine")
async def api_refine(
    feedback: str = Form(...),
    mode: str = Form("api"),
    size: str = Form("2048*2048"),
    negative_prompt: str = Form("low quality, blurry, distorted"),
    max_tokens: int = Form(300),
):
    """Refine the current image with user feedback."""
    global _session

    if not _session.get("current_prompt"):
        raise HTTPException(400, "No image generated yet. Generate one first.")
    if not feedback.strip():
        raise HTTPException(400, "Feedback is required")

    try:
        pipe = _get_pipeline()
        if _session.get("ref_image_paths"):
            try: pipe._load_preference_encoder_if_needed()
            except Exception: pass

        refined = pipe.refine_prompt_with_feedback(
            current_prompt=_session["current_prompt"], user_feedback=feedback.strip(),
            previous_image_path=_session.get("current_image_path"),
            image_paths=_session.get("ref_image_paths"), max_new_tokens=max_tokens,
        )

        result = pipe.generate_image_unified(
            prompt=refined,
            mode="api" if "api" in mode.lower() else "local",
            size=size, negative_prompt=negative_prompt, output_dir=str(OUTPUT_DIR),
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    if not result.get("success"):
        _unload()
        raise HTTPException(500, result.get("error", "Unknown error"))

    img_path = result.get("image_path", "")
    img_url = _img_path_to_url(img_path)
    pil_img = None
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        pass

    try:
        from evaluate_metrics import ImageGenerationEvaluator
        ev = ImageGenerationEvaluator()
        cs = ev.clip_score(refined, pil_img) if pil_img else 0
        aes = ev.aesthetic_score(pil_img) if pil_img else 0
    except Exception:
        cs = aes = 0

    turn_num = len(_session.get("turn_history", [])) + 1
    _session.setdefault("turn_history", []).append({
        "turn": turn_num, "feedback": feedback.strip(), "prompt": refined,
        "image_url": img_url, "clip_score": round(cs, 4), "aesthetic_score": round(aes, 4),
    })
    _session["current_prompt"] = refined
    _session["current_image_path"] = img_path

    return JSONResponse({
        "success": True,
        "image_url": img_url,
        "prompt": refined,
        "clip_score": round(cs, 4),
        "aesthetic_score": round(aes, 4),
        "turn": turn_num,
    })


@app.post("/api/reset")
async def api_reset():
    global _session
    _session = {"ref_image_paths": [], "turn_history": [], "current_prompt": "", "current_image_path": ""}
    return JSONResponse({"success": True})


@app.get("/api/history")
async def api_history():
    return JSONResponse(_session.get("turn_history", []))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
