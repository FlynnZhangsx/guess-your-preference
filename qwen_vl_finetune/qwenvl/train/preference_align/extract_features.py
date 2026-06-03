"""
Feature Pre-extraction + Offline Augmentation (v5)
----------------------------------------------------
Image augmentation (positive only): original + 4 transforms = 5 variants
Text augmentation: original + 2 variants (shuffled, dropped) = 3 variants

Saves:
  aug_text_features.pt  — list of [3, 512] per user
  aug_image_features.pt — dict, key -> [K, 512] (K=5 for pos, K=1 for neg)
  dataset_mapping.json  — same structure as before
"""

import os
import re
import json
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms as T

# ============================================================
# Configuration
# ============================================================

CSV_PATH = r"D:\code_vscode\多模态课设\realistic.csv"
IMAGE_ROOT = r"D:\QQfile\qwen_image"
OUTPUT_DIR = r"D:\code_vscode\多模态课设\Qwen3-VL\qwen_vl_finetune\qwenvl\train\preference_align\data_cache"

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cpu"
BATCH_SIZE = 16  # smaller batches since we extract multiple variants

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Image Augmentation Pipeline
# ============================================================

# Each transform applied independently to the original image
IMAGE_AUGS = {
    "flip":     T.RandomHorizontalFlip(p=1.0),
    "crop":     T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    "color":    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    "rotate":   T.RandomRotation(degrees=15),
}

def apply_image_augments(pil_image):
    """Return list of [original, flip_aug, crop_aug, color_aug, rotate_aug]."""
    # Original (resized to 224 for consistency with CLIP processing)
    orig = pil_image.resize((224, 224), Image.BICUBIC)
    results = [orig]
    for name in ["flip", "crop", "color", "rotate"]:
        results.append(IMAGE_AUGS[name](pil_image))
    return results  # 5 PIL images


# ============================================================
# Text Augmentation
# ============================================================

def create_text_variants(original_prompt):
    """
    Generate 3 text variants from the psychological prompt.

    V1: original
    V2: shuffle the clause order (split by period, reorder)
    V3: drop the last sentence (atmosphere/MBTI part)
    """
    variants = [original_prompt]

    # V2: shuffled sentence order
    sentences = [s.strip() for s in original_prompt.split('.') if s.strip()]
    if len(sentences) >= 3:
        rng = random.Random(hash(original_prompt) % (2**31))
        shuffled = sentences.copy()
        rng.shuffle(shuffled)
        variants.append('. '.join(shuffled) + '.')
    else:
        # Not enough sentences — duplicate original
        variants.append(original_prompt)

    # V3: drop last sentence (MBTI atmosphere)
    if len(sentences) >= 2:
        variants.append('. '.join(sentences[:-1]) + '.')
    else:
        variants.append(original_prompt)

    return variants  # list of 3 strings

# ============================================================
# Chinese-English Mapping Dict (same as before, abbreviated)
# ============================================================

SPACE_STYLE_MAP = {
    "温馨治愈风": "cozy healing atmosphere, warm lighting, soft textiles",
    "极简功能风": "minimalist functional design, clean lines, uncluttered space",
    "工业风": "industrial loft aesthetic, exposed brick, metal fixtures",
    "自然原木风": "natural wood tones, organic materials, earthy textures",
    "复古风": "vintage retro decor, nostalgic elements, antique finishes",
    "现代轻奢风": "modern luxury, marble surfaces, metallic gold accents",
    "波西米亚风": "bohemian eclectic, layered textiles, global patterns",
}

TRAVEL_MAP = {
    "城市探索": "urban exploration, cityscape views, metropolitan energy",
    "看展": "art gallery exhibitions, museum interiors, curated displays",
    "自然风光": "natural landscapes, sweeping vistas, verdant scenery",
    "徒步": "hiking trails, mountain paths, adventurous outdoors",
    "海岛 / 沙滩": "tropical beach, ocean waves, seaside serenity",
    "历史古迹": "ancient historical sites, weathered stone, timeless monuments",
    "文化游": "cultural immersion, traditional architecture, heritage details",
}

FASHION_MAP = {
    "运动休闲": "athleisure sporty style, casual comfortable fabrics",
    "简约风": "minimalist fashion, neutral tones, clean silhouettes",
    "通勤风": "smart professional attire, tailored office wear",
    "街头潮流": "streetwear urban fashion, bold graphics, contemporary edge",
    "暗黑风": "dark gothic aesthetic, black leather, dramatic silhouettes",
    "可爱风": "cute playful style, pastel accents, charming details",
}

HOBBY_MAP = {
    "影视剧": "cinematic scenes, film-inspired visuals, dramatic storytelling",
    "动漫": "anime aesthetics, illustrated characters, vibrant cel-shading",
    "漫画": "comic book art, graphic novel panels, bold linework",
    "游戏": "video game concept art, immersive virtual worlds, pixel motifs",
    "音乐": "musical instruments, melodic visual rhythms, concert atmosphere",
    "写作": "literary motifs, ink and paper textures, poetic calligraphy",
    "阅读": "library aesthetics, book pages, quiet contemplative mood",
    "摄影": "photographic compositions, lens flare, expertly framed shots",
    "手工": "handcrafted artisan textures, DIY aesthetic, tactile materials",
    "探索自然": "nature exploration, botanical illustrations, wildlife motifs",
    "与宠物亲密互动": "adorable pet companions, cozy animal moments, warm bond",
    "品尝美食": "culinary food photography, gourmet plating, appetizing colors",
}

SCHWARTZ_VISUAL_MAP = {
    "自主独立：独立 / 创造力": "independent, creative, self-directed expression",
    "刺激体验：新奇 / 兴奋": "dynamic, vibrant, experimental, bold adventure",
    "享乐主义：愉悦": "pleasure-oriented, indulgent, sensuous enjoyment",
    "成就追求：个人成功": "aspirational, achievement-focused, triumphant success",
    "权力掌控：地位 / 掌控力": "powerful, dominant, commanding presence",
    "安全稳定：安全 / 稳定": "peaceful, well-balanced proportions, serene stability",
    "循规守矩：遵守规范": "orderly, conventional, rule-following structure",
    "传统尊重：尊崇习俗": "traditional, classic, time-honored customs",
    "仁爱利他：帮助他人": "warm, altruistic, compassionate, helping hands",
    "普世关怀：包容 / 关爱自然": "nature-oriented, harmonious, universal compassion",
}

COLOR_MAP = {
    "温暖金色": "warm golden hues, amber sunlight, rich brass tones",
    "梦幻马卡龙": "pastel macaron color palette, soft candy hues, dreamy confectionery shades",
    "秋日风情": "autumnal harvest colors, burnt orange, deep burgundy, golden wheat",
    "都市冷淡": "urban cool-tone grays, muted concrete neutrals, steel blue accents",
    "活力霓虹": "vibrant neon electric, fluorescent brights, cyberpunk glow",
    "柔和大地": "soft earthy tones, terracotta clay, sage green, warm sand beige",
    "清凉海洋": "cool oceanic blues, teal, aquamarine, crisp sea foam",
    "高对比度黑白": "high-contrast black and white, stark chiaroscuro, dramatic monochrome",
}

def get_mbti_vibe(mbti_str):
    mbti = mbti_str.strip().upper()
    if any(t in mbti for t in ["INTJ", "INTP", "ENTJ", "ENTP"]):
        return "structured, minimalist, intellectually refined, clean geometric precision"
    elif any(t in mbti for t in ["INFJ", "INFP", "ENFJ", "ENFP"]):
        return "dreamy, emotionally expressive, poetic, soft ethereal glow"
    elif any(t in mbti for t in ["ISTJ", "ISFJ", "ESTJ", "ESFJ"]):
        return "classic, neat, orderly, traditional elegance, well-organized"
    elif any(t in mbti for t in ["ISTP", "ISFP", "ESTP", "ESFP"]):
        return "bold, sensory-rich, spontaneous, dynamic energy, adventurous spirit"
    else:
        return "balanced, versatile, eclectic, thoughtfully composed"

# ============================================================
# 1. Read CSV & Build Prompts
# ============================================================

print("=" * 60)
print("[1/6] Reading CSV and building psychological prompts...")
print("=" * 60)

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df = df[df.iloc[:, 0].notna()]
print(f"  Valid rows: {len(df)}")

def select_one(s):
    if pd.isna(s) or str(s).strip() == '':
        return ''
    return re.sub(r'^[A-Z]\s*[.．]\s*', '', str(s).strip())

def get_numeric(row, col_idx, default=4):
    try:
        val = float(row.iloc[col_idx])
        return int(val) if not pd.isna(val) else default
    except:
        return default

def build_english_prompt(row):
    q12 = get_numeric(row, 61)
    q13 = get_numeric(row, 62)
    if q12 >= 4 and q13 <= 3:
        composition = "abstract, surrealist elements, non-traditional asymmetrical layout, unconventional perspective, creative rule-breaking design"
    elif q13 >= 5 or q12 <= 3:
        composition = "figurative, realistic photography, highly symmetrical balanced composition, classic centered framing, traditional poster layout"
    else:
        composition = "balanced blend of abstract and realistic elements, moderately creative composition with subtle asymmetry"

    value_visuals = []
    for i in range(65, 75):
        v = select_one(row.iloc[i])
        if v and v in SCHWARTZ_VISUAL_MAP:
            value_visuals.append(SCHWARTZ_VISUAL_MAP[v])
    dynamics = ", ".join(value_visuals[:3]) if value_visuals else "balanced, moderate energy, universally appealing"
    if any("刺激体验" in str(row.iloc[i]) for i in range(65, 75)):
        dynamics = "dynamic, vibrant, experimental, high-energy visual impact, " + dynamics
    if any("普世关怀" in str(row.iloc[i]) for i in range(65, 75)):
        dynamics += ", nature-oriented, harmonious unity, inclusive diversity"

    q14 = get_numeric(row, 63)
    q15 = get_numeric(row, 64)
    arousal_part = "high saturation, strong contrast, vivid energetic lighting, bold color intensity" if q15 >= 4 else \
                   "low saturation, misty lighting, minimalist muted tones, soft diffused light" if q15 <= 3 else \
                   "moderate saturation, balanced contrast, natural ambient lighting"
    valence_part = "bright, uplifting mood, cheerful radiant atmosphere, warm inviting glow" if q14 >= 4 else \
                   "moody, subdued atmospheric tones, melancholic depth, introspective shadows" if q14 <= 3 else \
                   "neutral balanced mood, calm composed emotional tone"
    arousal_valence = f"{arousal_part}, {valence_part}"

    color_visuals = []
    for i in range(76, 84):
        c = select_one(row.iloc[i])
        if c and c in COLOR_MAP:
            color_visuals.append(COLOR_MAP[c])
    colors = ", ".join(color_visuals[:3]) if color_visuals else "versatile balanced color palette"

    space = select_one(row.iloc[12])
    space_visual = SPACE_STYLE_MAP.get(space, "inviting interior atmosphere") if space else "inviting interior atmosphere"

    travel_visuals = [TRAVEL_MAP[select_one(row.iloc[i])] for i in range(13, 21) if select_one(row.iloc[i]) in TRAVEL_MAP]
    vibe_travel = f"{space_visual}, " + ", ".join(travel_visuals[:2]) if travel_visuals else space_visual

    hobby_visuals = [HOBBY_MAP[select_one(row.iloc[i])] for i in range(28, 41) if select_one(row.iloc[i]) in HOBBY_MAP]
    fashion_visuals = [FASHION_MAP[select_one(row.iloc[i])] for i in range(21, 28) if select_one(row.iloc[i]) in FASHION_MAP]
    hobbies_fashion = ", ".join((hobby_visuals[:2] + fashion_visuals[:1])) if (hobby_visuals or fashion_visuals) else "lifestyle motifs, everyday aesthetic details"

    mbti_str = str(row.iloc[59]).strip()
    mbti_vibe = get_mbti_vibe(mbti_str) if (mbti_str and mbti_str.upper() != 'NAN' and mbti_str) else "balanced, versatile, eclectic"

    prompt = (
        f"An aesthetically pleasing poster design featuring {composition}, "
        f"with a {dynamics} visual scheme. "
        f"The lighting and colors are {arousal_valence}, "
        f"using a color palette of {colors}. "
        f"The setting evokes {vibe_travel}, "
        f"incorporating motifs of {hobbies_fashion}. "
        f"Overall atmosphere: {mbti_vibe}."
    )
    return prompt

user_profiles = {}
for idx, row in df.iterrows():
    user_id = str(int(row.iloc[0]))
    prompt = build_english_prompt(row)
    if prompt:
        user_profiles[user_id] = prompt
print(f"  Built prompts for {len(user_profiles)} users")

# ============================================================
# 2. Scan Images
# ============================================================

print("\n[2/6] Scanning image directories...")
def parse_info_txt(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    for d in ['，', ',', '、', '\n', '\r']:
        content = content.replace(d, ' ')
    return [int(t) for t in content.split() if t.strip().isdigit()]

def get_image_files(fp):
    return [f for f in os.listdir(fp) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

def resolve_image_path(folder, idx):
    for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
        p = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(p):
            return p
    return None

dataset_records = []
positive_image_paths = set()
all_image_paths_list = []

all_dirs = sorted([d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d)) and d.isdigit()], key=lambda x: int(x))

for dir_name in tqdm(all_dirs, desc="  Scanning"):
    folder = os.path.join(IMAGE_ROOT, dir_name)
    info_path = os.path.join(folder, "info.txt")
    if not os.path.exists(info_path):
        continue
    pos_indices = parse_info_txt(info_path)
    if not pos_indices:
        continue
    if dir_name not in user_profiles:
        continue

    all_imgs = get_image_files(folder)
    if len(all_imgs) < 2:
        continue

    pos_paths = []
    for idx in pos_indices:
        p = resolve_image_path(folder, idx)
        if p:
            pos_paths.append(p)
            positive_image_paths.add(p)

    pos_basenames = set(os.path.basename(p) for p in pos_paths)
    neg_paths = [os.path.join(folder, f) for f in all_imgs if f not in pos_basenames]

    if not pos_paths or not neg_paths:
        continue

    for p in pos_paths + neg_paths:
        all_image_paths_list.append(p)

    dataset_records.append({
        'user_id': dir_name, 'text': user_profiles[dir_name],
        'pos_image_paths': pos_paths, 'neg_image_paths': neg_paths,
    })

print(f"  Users: {len(dataset_records)}, Pos images: {len(positive_image_paths)}, Total images: {len(set(all_image_paths_list))}")

# ============================================================
# 3. Load CLIP
# ============================================================

print("\n[3/6] Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
print(f"  CLIP loaded on {DEVICE}")

def extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unknown output type: {type(output)}")

# ============================================================
# 4. Extract Augmented TEXT Features
# ============================================================

print("\n[4/6] Extracting augmented TEXT features (3 variants/user)...")
aug_text_features = []  # list of [3, 512]

# Precompute all text variants
all_text_variants = []   # flat list for batch extraction
user_variant_map = []    # (user_idx, variant_count) — we'll reconstruct later

for rec in dataset_records:
    variants = create_text_variants(rec['text'])
    start = len(all_text_variants)
    all_text_variants.extend(variants)
    user_variant_map.append((start, len(variants)))

all_text_feats = []
with torch.no_grad():
    for i in tqdm(range(0, len(all_text_variants), BATCH_SIZE), desc="  Text"):
        batch = all_text_variants[i:i+BATCH_SIZE]
        inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        emb = clip_model.get_text_features(**inputs)
        emb = extract_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_text_feats.append(emb.cpu())

all_text_feats = torch.cat(all_text_feats, dim=0)  # [total_variants, 512]

# Reconstruct per-user [3, 512]
for start, count in user_variant_map:
    aug_text_features.append(all_text_feats[start:start+count])  # [3, 512]
print(f"  Augmented text features: {len(aug_text_features)} users x {aug_text_features[0].size(0)} variants")
print(f"  Shape per user: {aug_text_features[0].shape}")

# ============================================================
# 5. Extract Augmented IMAGE Features
# ============================================================

print("\n[5/6] Extracting augmented IMAGE features (5 variants for pos, 1 for neg)...")

# Build path-to-key mapping
image_path_to_key = {}
for p in set(all_image_paths_list):
    key = os.path.relpath(p, IMAGE_ROOT).replace('\\', '/')
    image_path_to_key[p] = key

# Separate pos and neg paths
neg_image_paths = [p for p in set(all_image_paths_list) if p not in positive_image_paths]

aug_image_features = {}

# --- Negative images: single feature ---
print(f"  Negatives: {len(neg_image_paths)} images (1 variant each)")
with torch.no_grad():
    for i in tqdm(range(0, len(neg_image_paths), BATCH_SIZE), desc="  Neg images"):
        batch_paths = neg_image_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
            except:
                batch_imgs.append(Image.new("RGB", (224, 224)))

        inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
        emb = clip_model.get_image_features(**inputs)
        emb = extract_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        for j, e in enumerate(emb.cpu()):
            key = image_path_to_key[batch_paths[j]]
            aug_image_features[key] = e.unsqueeze(0)  # [1, 512]

# --- Positive images: 5 variants each ---
pos_image_path_list = list(positive_image_paths)
print(f"  Positives: {len(pos_image_path_list)} images (5 variants each = {len(pos_image_path_list)*5} total)")

import time as _time
_start_time = _time.time()
with torch.no_grad():
    pbar = tqdm(pos_image_path_list, desc="  Pos images")
    for i, path in enumerate(pbar):
        try:
            pil_img = Image.open(path).convert("RGB")
        except:
            pil_img = Image.new("RGB", (224, 224))

        # 5 variants: original + 4 transforms
        aug_images = apply_image_augments(pil_img)

        # Batch-extract all 5
        inputs = clip_processor(images=aug_images, return_tensors="pt").to(DEVICE)
        emb = clip_model.get_image_features(**inputs)
        emb = extract_tensor(emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # [5, 768]

        key = image_path_to_key[path]
        aug_image_features[key] = emb.cpu()

        # Detailed logging every 50 images
        if (i + 1) % 50 == 0:
            elapsed = _time.time() - _start_time
            rate = (i + 1) / elapsed
            eta = (len(pos_image_path_list) - i - 1) / rate
            pbar.write(f"  [Pos] Processed {i+1}/{len(pos_image_path_list)} images "
                       f"({rate:.1f} img/s, ETA: {eta/60:.1f} min)")

print(f"  Augmented image features: {len(aug_image_features)} keys")
pos_example = list(positive_image_paths)[0]
print(f"  Pos shape: {aug_image_features[image_path_to_key[pos_example]].shape}")
neg_example = neg_image_paths[0]
print(f"  Neg shape: {aug_image_features[image_path_to_key[neg_example]].shape}")

# ============================================================
# 6. Save
# ============================================================

print("\n[6/6] Saving...")

# Save augmented text features (list of tensors)
torch.save(aug_text_features, os.path.join(OUTPUT_DIR, "aug_text_features.pt"))
print(f"  aug_text_features.pt: {len(aug_text_features)} users")

# Save augmented image features dict
torch.save(aug_image_features, os.path.join(OUTPUT_DIR, "aug_image_features.pt"))
print(f"  aug_image_features.pt: {len(aug_image_features)} images")

# Reuse existing dataset_mapping (same structure)
dataset_mapping = []
for idx, rec in enumerate(dataset_records):
    pos_keys = [image_path_to_key[p] for p in rec['pos_image_paths'] if p in image_path_to_key]
    neg_keys = [image_path_to_key[p] for p in rec['neg_image_paths'] if p in image_path_to_key]
    if not pos_keys or not neg_keys:
        continue
    dataset_mapping.append({
        "user_id": rec['user_id'], "text_idx": idx,
        "pos_image_keys": pos_keys, "neg_image_keys": neg_keys,
    })

with open(os.path.join(OUTPUT_DIR, "dataset_mapping.json"), 'w', encoding='utf-8') as f:
    json.dump(dataset_mapping, f, ensure_ascii=False, indent=2)
print(f"  dataset_mapping.json: {len(dataset_mapping)} users")

print("\n" + "=" * 60)
print("Augmented feature extraction complete!")
print("=" * 60)
