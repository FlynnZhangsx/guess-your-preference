# Cross-Modal User Preference Alignment Module

用户画像（文本）与偏好参考图（图像）的跨模态偏好对齐模块。

将用户的问卷数据和他们喜欢的图片映射到统一的 256 维偏好空间（Preference Space）中，通过 Triplet Ranking 学习使文本向量与偏好图片向量在空间中靠近，与不喜欢图片向量远离。

## 架构

```
问卷文本 ──→ CLIP Text Encoder ──→ [512] ──→ Text_MLP ──→ [256] ┐
                                                                   ├──→ Preference Space (余弦相似度)
参考图像 ──→ CLIP Image Encoder ──→ [512] ──→ Image_MLP ──→ [256] ┘
```

- **特征提取**：`openai/clip-vit-base-patch32`（离线预提取，纯 CPU 友好）
- **投影网络**：两个轻量 MLP（512 → 256 → 256，ReLU + Dropout + L2 Norm）
- **损失函数**：`TripletMarginLoss`（margin=0.3）
- **参数量**：约 525K（极其轻量）

## 目录结构

```
preference_align/
├── extract_features.py    # Step 1: 离线特征提取
├── model.py               # Step 2a: MLP 投影网络定义
├── dataset.py             # Step 2b: Triplet Dataset
├── train.py               # Step 3: 训练 + 评估
├── data_cache/            # [自动生成] 特征缓存
│   ├── text_features.pt   # 文本 CLIP 特征
│   ├── image_features.pt  # 图像 CLIP 特征 (dict)
│   └── dataset_mapping.json  # 用户-正负样本映射
├── preference_model.pth   # [自动生成] 最终模型权重
├── preference_model_best.pth  # [自动生成] 最佳模型权重
└── README.md
```

## 运行顺序

### Step 1: 特征预提取

```bash
cd preference_align/
python extract_features.py
```

**功能**：
- 读取 `D:\code_vscode\多模态课设\realistic.csv`（用户问卷数据）
- 扫描 `D:\QQfile\qwen_image`（用户偏好图片）
- 解析 `info.txt`（正样本标注）并过滤无效数据
- 使用 CLIP 提取所有文本和图像的特征向量
- 生成 `data_cache/` 目录下的缓存文件

**预估耗时**（纯 CPU）：约 5-15 分钟（取决于 CPU 核心数）

---

### Step 2: 训练投影网络

```bash
python train.py
```

**功能**：
- 加载缓存特征
- 80/20 划分为训练/验证集
- 使用 TripletMarginLoss 训练 MLP 投影网络
- 每 epoch 计算 Retrieval Accuracy（正样本余弦相似度 > 负样本的概率）
- 自动保存最优模型和最终模型

**配置**（可在 `train.py` 顶部修改）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LEARNING_RATE` | 1e-4 | Adam 学习率 |
| `BATCH_SIZE` | 128 | 批次大小 |
| `NUM_EPOCHS` | 80 | 训练轮数 |
| `TRIPLET_MARGIN` | 0.3 | Triplet Loss 间隔 |
| `TRAIN_SPLIT` | 0.8 | 训练集比例 |

**预估耗时**（纯 CPU）：约 2-5 分钟

---

## 评估指标

- **Retrieval Accuracy**：对于每个用户，随机采样正样本和负样本，计算模型投影后的余弦相似度。如果 `sim(text, positive) > sim(text, negative)` 则视为命中。

## 模型推理示例

```python
import torch
from model import PreferenceAlignModel

model = PreferenceAlignModel()
checkpoint = torch.load("preference_model_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# text_feat: [512] CLIP text feature
# image_feat: [512] CLIP image feature
text_proj = model.encode_text(text_feat.unsqueeze(0))    # [1, 256]
image_proj = model.encode_image(image_feat.unsqueeze(0)) # [1, 256]

similarity = (text_proj * image_proj).sum(dim=-1)  # cosine similarity
```
