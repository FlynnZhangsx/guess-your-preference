# Cross-Modal User Preference Alignment — 推理管线

用户输入简短提示词 + 0~N 张参考图片 → 偏好对齐模块 → Qwen3-VL 丰富 prompt → Qwen-Image 生成海报。

## 管线架构

```
User Images (0~N)  +  简短文本请求
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
CLIP Text Encoder               │
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
```

### 模块说明

| 模块 | 模型 | 说明 |
|------|------|------|
| 偏好编码器 | `openai/clip-vit-large-patch14` + `Image_MLP` | 冻结 CLIP + 已训练的 MLP 残差投影（Val AUC=0.8934） |
| 风格匹配 | CLIP Text Encoder | 零样本余弦相似度匹配 27 种美学风格描述符 |
| Prompt 增强 | `Qwen3-VL-4B-Instruct` | 融合风格描述 + 用户请求，输出英文 T2I prompt |
| 图像生成 | `Qwen-Image-2512` (本地) 或 `qwen-image-2.0-pro` (API) | 支持两种模式 |

---

## 目录结构

```
preference_align/
├── pipeline.py                 # 🔥 主推理管线（支持 local/API 双模式）
├── model.py                    # 偏好对齐模型定义（Image_MLP / Text_MLP）
├── qwen_bridge.py              # 虚拟 Token 投影（768 → Qwen-VL embedding）
├── train.py                    # 训练脚本
├── dataset.py                  # Triplet Dataset
├── extract_features.py         # CLIP 特征预提取
├── preference_model_best.pth   # 最优模型权重（~14MB）
├── preference_model_v7.pth     # V7 模型权重（~4.7MB）
├── output/                     # 生成图片输出目录
└── README.md
```

---

## 快速开始

### 环境准备

```bash
# 激活虚拟环境
source /home/coder/project/data/mllm/duomotai/mllm/bin/activate

# 安装依赖（如未安装）
pip install dashscope pillow
```

### 方式一：API 模式（推荐，无需 GPU 加载 Qwen-Image）

**优点**：不需要 ~24GB GPU 显存加载 Qwen-Image，仅需 Qwen3-VL 做 prompt 增强。

**配置 API Key**：

代码中已预置 API Key（`pipeline.py` 顶部 `DASHSCOPE_API_KEY`），也可通过环境变量覆盖：

```bash
export DASHSCOPE_API_KEY="sk-xxx"
```

获取 Key：https://help.aliyun.com/zh/model-studio/get-api-key

**运行**：

```bash
cd preference_align/

# 0 张参考图（使用中性偏好质心）
python pipeline.py --api 0img

# 1 张参考图（偏好对齐）
python pipeline.py --api 1img

# Demo（模拟参考图）
python pipeline.py --api demo
```

### 方式二：本地模式（需要 GPU ~24GB VRAM）

```bash
# 0 张参考图
python pipeline.py 0img

# 1 张参考图
python pipeline.py 1img
```

---

## Python API 调用

```python
from pipeline import PersonalizationPipeline

# 初始化（加载 Qwen3-VL）
pipe = PersonalizationPipeline(
    qwen_model_name="/path/to/Qwen3-VL-4B-Instruct",
    load_qwen_vl=True,
)

# Step 1: 偏好对齐 → 丰富 prompt
enriched = pipe.generate_personalized_prompt(
    image_paths=["/path/to/ref.jpg"],  # 或 [] 使用中性偏好
    short_prompt="一只小猫坐在月光下的窗台上",
)

# Step 2: 卸载 Qwen3-VL + 偏好编码器（释放显存）
pipe.unload_preference_encoder()
pipe.unload_qwen_vl()

# Step 3: 生成图像

# API 模式
result = pipe.generate_image_unified(
    enriched,
    mode="api",
    size="2048*2048",           # 输出尺寸
    api_model="qwen-image-2.0-pro",
)

# 本地模式
result = pipe.generate_image_unified(
    enriched,
    mode="local",
    width=1664, height=928,      # 16:9
    num_inference_steps=50,
    seed=42,
)

print(result)  # {"success": True, "image_path": "./output/poster_xxx.png", ...}
```

---

## 关键参数

### `generate_personalized_prompt()`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_paths` | `List[str]` | — | 参考图路径列表，`[]` 表示使用中性偏好 |
| `short_prompt` | `str` | — | 用户简短文本请求 |
| `max_new_tokens` | `int` | `300` | Qwen3-VL 最大生成长度 |

### `generate_image_unified()` — API 模式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `str` | `"local"` | `"local"` 或 `"api"` |
| `size` | `str` | `"2048*2048"` | 输出尺寸 |
| `api_model` | `str` | `"qwen-image-2.0-pro"` | API 模型名 |
| `prompt_extend` | `bool` | `True` | DashScope 内置 prompt 扩展 |
| `watermark` | `bool` | `False` | 添加水印 |
| `api_key` | `str` | `None` | API Key（可选，默认用代码内置） |

### `generate_image_unified()` — 本地模式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `width` | `int` | `1664` | 输出宽度 |
| `height` | `int` | `928` | 输出高度 |
| `num_inference_steps` | `int` | `50` | 扩散步数 |
| `true_cfg_scale` | `float` | `4.0` | CFG 引导强度 |
| `seed` | `int` | `42` | 随机种子 |
| `qwen_image_model` | `str` | `"Qwen/Qwen-Image-2512"` | HF 模型 ID |

---

## 偏好对齐原理

### 0 张参考图（中性偏好）

无用户参考图时，使用所有 27 个风格描述符在 CLIP 空间中的**质心向量**作为中性偏好。这等价于"平衡"所有风格的一个折中方案。

### N 张参考图（用户偏好对齐）

1. **提取**：每张图通过 CLIP ViT-L/14 → Image_MLP → L2-norm → [768] 偏好向量
2. **融合**：N 个向量取均值 → L2-norm → 最终偏好向量
3. **匹配**：余弦相似度 Top-5 匹配 27 个美学风格描述符
4. **注入**：Qwen3-VL 将风格描述拼入 system prompt，生成带风格的 T2I prompt

### 27 种美学风格描述符

覆盖 5 个维度：构图（4）、色彩（7）、光影/氛围（4）、风格/质地（6）、情绪/气氛（6）

---

## CLI 命令

```bash
# API 模式
python pipeline.py --api 0img    # 0图 + API
python pipeline.py --api 1img    # 1图 + API
python pipeline.py --api demo    # 模拟demo + API

# 本地模式（默认）
python pipeline.py 0img          # 0图 + 本地
python pipeline.py 1img          # 1图 + 本地
python pipeline.py demo          # 模拟demo + 本地

# 无参数默认 = 0图 + 本地
python pipeline.py
```

---

## 训练相关

偏好对齐模块的训练不在此 README 范围内，详见训练脚本：

```bash
# Step 1: 特征提取
python extract_features.py

# Step 2: 训练
python train.py
```

模型权重：
- `preference_model_best.pth` — 最优模型（Val AUC=0.8934），用于推理
- `preference_model_v7.pth` — V7 版本权重

---

## 生成示例

| 模式 | 输入 | 输出 |
|------|------|------|
| 0-image (中性偏好) | `一只小猫坐在月光下的窗台上` | [output/poster_99901.png](output/poster_99901.png) |
| 0-image (旧版无偏好) | 同上 | [output/poster_39578.png](output/poster_39578.png) |
| 1-image (蓝色参考图) | 同上 + 蓝色参考图 | [output/poster_73333.png](output/poster_73333.png) |

---

## 实验：三种策略对比

`run_experiments.py` 自动对每个测试用例跑三种 prompt 策略并生成图片。

### 三种模式

| 模式 | 名称 | 原理 |
|------|------|------|
| **A** | Zero-shot Baseline | 短提示词直接送 Qwen-VL 扩写，无任何偏好注入 |
| **B** | Hard Prompting (Text Style) | 从参考图提取偏好 → 匹配 Top-5 风格文字 → 拼入 system prompt |
| **C** | Soft Prompting (Virtual Tokens) | 从参考图提取偏好 → PreferenceProjector → K 个连续虚拟 token 注入 inputs_embeds |

### 架构对比

```
Mode A:  short_prompt ──→ Qwen-VL ──→ enriched ──→ Qwen-Image

Mode B:  ref_images → CLIP+MLP → style text ──┐
                                               ├──→ Qwen-VL ──→ Qwen-Image
         short_prompt ─────────────────────────┘

Mode C:  ref_images → CLIP+MLP → pref_vec → PreferenceProjector → virtual tokens
                                                                    │
         short_prompt ──→ text embeddings ──→ [virtual_tokens | text_embeds] → Qwen-VL → Qwen-Image
```

### 运行

```bash
# 1. 编辑 run_experiments.py 中的 TEST_CASES，填入参考图路径
# 2. 运行全部
python run_experiments.py

# 只跑指定模式
python run_experiments.py --mode A          # 只跑 Baseline
python run_experiments.py --mode B,C        # 只跑 B 和 C

# 只跑指定测试用例
python run_experiments.py --cases 0         # 只跑第一个用例
python run_experiments.py --cases 0,1       # 跑前两个
```

输出结构：

```
experiment_output/
├── cat_moonlight/
│   ├── cat_moonlight_mode_A.png
│   ├── cat_moonlight_mode_B.png
│   └── cat_moonlight_mode_C.png
├── summer_poster/
│   └── ...
└── experiment_results.json        # 包含所有 enriched prompt 和结果
```

### 准备测试数据

在 `run_experiments.py` 的 `TEST_CASES` 列表中定义用例：

```python
TEST_CASES = [
    TestCase(
        id="cat_moonlight",
        short_prompt="一只小猫坐在月光下的窗台上",
        ref_images=["/path/to/ref1.jpg", "/path/to/ref2.jpg"],
        description="Cat on a moonlit windowsill",
    ),
    # 添加更多用例...
]
```

---

## 评估：CLIP 自动指标

`evaluate_metrics.py` 用 CLIP ViT-B/32 计算客观指标。

### 两项指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Text↔Image** | `cos_sim(CLIP(short_prompt), CLIP(gen_image))` | 基础语义符合度 — 图片是否捕捉了用户请求 |
| **Image↔Image** | `cos_sim(mean(CLIP(ref_images)), CLIP(gen_image))` | 视觉偏好一致性 — 输出是否像参考图的风格 |

### 运行

```bash
# 自动扫描 experiment_output/ 目录
python evaluate_metrics.py

# 指定目录和参考图
python evaluate_metrics.py --dir ./experiment_output --ref-dir ./my_refs

# 输出 JSON 报告
python evaluate_metrics.py --output ./my_report.json
```

输出示例：

```
PER-CASE RESULTS
  ▸ [cat_moonlight] "一只小猫坐在月光下的窗台上"
    Mode                                     Text↔Image    Image↔Image
    A: Zero-shot Baseline                      0.3124          N/A
    B: Hard Prompting (Text Style)             0.3281         0.5418
    C: Soft Prompting (Virtual Tokens)         0.3412         0.5633

AGGREGATE RESULTS (mean ± std across 3 cases)
  Mode                                     Text↔Image        Image↔Image
  A: Zero-shot Baseline                  0.3056 ± 0.0123        N/A
  B: Hard Prompting (Text Style)         0.3215 ± 0.0156    0.5321 ± 0.0221
  C: Soft Prompting (Virtual Tokens)     0.3389 ± 0.0189    0.5517 ± 0.0198

DELTA FROM BASELINE (Mode A = Zero-shot)
  Mode                                      Δ Text↔Image    Winner
  B: Hard Prompting vs A                      +0.0159       ✓ BETTER
  C: Soft Prompting vs A                      +0.0333       ✓ BETTER

HEAD-TO-HEAD: Mode C (Soft) vs Mode B (Hard)
  Δ Text↔Image (C - B): +0.0174  — C wins ✓
```

### 人工打分

Web UI (`webui.py`) 的 Evaluate Tab 提供 1-5 分人工评分界面，评分保存至 `human_ratings/ratings_YYYYMMDD.jsonl`。

---

## 依赖

- `torch` >= 2.0
- `transformers` >= 4.45
- `diffusers`（本地模式）
- `dashscope`（API 模式）
- `gradio`（Web UI）
- `Pillow`
- `accelerate`（可选，自动 device_map）
