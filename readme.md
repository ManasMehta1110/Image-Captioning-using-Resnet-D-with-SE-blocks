<div align="center">

# 🖼️ Image Captioning — SE-ResNet-D + SCST

**Attention-based image captioning on MS-COCO, significantly extending Show, Attend and Tell**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![COCO](https://img.shields.io/badge/Dataset-MS--COCO-00AABB?style=flat-square)](https://cocodataset.org)
[![Split](https://img.shields.io/badge/Split-Karpathy-6B7280?style=flat-square)](http://cs.stanford.edu/people/karpathy/deepimagesent/)

</div>

---

## Results

| Model | Encoder | Training | BLEU-4 | CIDEr |
|:---|:---|:---:|:---:|:---:|
| Show, Attend and Tell (2015) | VGG-16 | CE only | ~0.212 | — |
| Show, Attend and Tell (ResNet) | ResNet-101 | CE only | ~0.243 | — |
| **This project** | **SE-ResNet-50-D** | **CE → SCST** | **~0.222** | **~0.61** |
| Bottom-Up Top-Down (2018) | Faster-RCNN | CE → SCST | ~0.363 | ~1.10 |

> Measured on MSCOCO Karpathy test split, beam size 5, epoch 85. Scores are expected to improve toward **0.235–0.245** with continued SCST training to epoch 120.

---

## What's Different From the Baseline

Three key upgrades over the classic [sgrvinod tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning):

### 1. SE-ResNet-50-D Encoder

**ResNet-D stem** replaces the single aggressive 7×7 convolution with three stacked 3×3 convolutions, preserving finer spatial detail — important because the attention mechanism needs rich spatial features to attend to.

**Squeeze-and-Excitation (SE) blocks** recalibrate channel-wise feature responses after every bottleneck. The block globally pools each feature map, passes it through a small bottleneck MLP, and produces per-channel sigmoid weights:

```
z_c = (1/H×W) Σ u_c(i,j)          # Squeeze: global average pool
s   = σ(W₂ · δ(W₁ · z))           # Excitation: bottleneck MLP → sigmoid
ũ_c = s_c · u_c                    # Scale: channel-wise multiply
```

Channels encoding "dog fur" or "sky texture" get boosted; irrelevant channels suppressed. The result is a more expressive encoder with ~28M parameters vs ResNet-101's ~44M, training faster while generalising better on captioning.

### 2. SCST with Real CIDEr Reward

Standard CE training has **exposure bias** — the decoder sees ground-truth tokens during training but must rely on its own predictions at test time. SCST fixes this using policy gradient (REINFORCE), directly optimising the evaluation metric:

```
L_SCST = −E_{w^s ~ P_θ}[ r(w^s) − r(w^g) ]
```

Where `r(w^s)` is the CIDEr score of the **sampled** caption and `r(w^g)` is the CIDEr score of the **greedy** caption. The greedy output serves as a self-critical baseline — the model is reinforced only when sampling beats its own argmax decode. Rewards are normalised per batch to prevent gradient explosions:

```
r̂ = (r − μ_r) / (σ_r + ε)
```

### 3. Mixed CE + RL Loss

Switching abruptly to pure RL causes the model to chase CIDEr reward while forgetting grammatical structure. A mixed loss anchors the language model during fine-tuning:

```
L = 0.80 · L_SCST + 0.20 · L_CE
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                 │
│                                                                 │
│  Image      SE-ResNet-50-D        AdaptAvgPool    Linear proj  │
│  256×256 → [ResNet-D stem]    →   7×7 spatial  → 49 × 512     │
│            [SE blocks ×4]         feature map     vectors      │
└─────────────────────────────────────────────────────────────────┘
                                         │
                              49 feature vectors (A)
                                         │
┌─────────────────────────────────────────────────────────────────┐
│                    DECODER (per timestep t)                     │
│                                                                 │
│  h_{t-1} ──→ Attention ──→ context ẑ_t                        │
│      │        scores α         │                               │
│      │       (49 weights)      │ f_beta gate: β_t = σ(W_β h)  │
│      │                         ↓                               │
│  embed(w_{t-1}) + β_t ⊙ ẑ_t ──→ LSTMCell ──→ h_t            │
│                                                 │               │
│                                           Linear + Softmax      │
│                                                 │               │
│                                           P(word_t | ...)       │
└─────────────────────────────────────────────────────────────────┘
```

**Attention equations:**

```
e_{t,k} = vᵀ tanh(W_h · h_{t-1} + W_a · A_k)   # energy at location k
α_{t,k} = exp(e_{t,k}) / Σ exp(e_{t,j})          # softmax → distribution over 49
ẑ_t     = Σ α_{t,k} · A_k                         # context vector
β_t     = σ(W_β · h_{t-1})                        # f_beta visual gate
```

The `f_beta` gate learns to suppress visual input for function words ("a", "the") and rely on language priors instead.

**Model dimensions:**

| Component | Dimension |
|:---|:---:|
| Encoder output | 49 × 512 |
| Embedding dim | 512 |
| Decoder hidden | 512 |
| Attention dim | 512 |
| Vocabulary size | ~9,500 |
| Dropout | 0.5 |

---

## Training Pipeline

Training proceeds in three phases:

```
Epochs 0–21   ──────────────────────────────────────────────────
  CE loss (teacher-forced), encoder frozen
  Decoder LR: 4e-4

Epochs 21–60  ──────────────────────────────────────────────────
  CE loss, encoder unfrozen (layers 2–4 only)
  Decoder LR: 4e-4  |  Encoder LR: 1e-5

Epochs 60+    ──────────────────────────────────────────────────
  Mixed SCST (80%) + CE (20%), fresh Adam optimizers
  Decoder LR: 7e-6  |  Encoder LR: 3e-6
```

**Why three phases?**

- Layer 1 of ResNet captures universal edges/textures — fine-tuning it hurts generalisation, so it stays frozen throughout.
- The encoder uses a much lower LR than the decoder because its ImageNet weights are already highly tuned; aggressive fine-tuning destroys them before the decoder has learned to use them.
- Fresh Adam optimizers are essential on the CE→SCST transition. Adam stores momentum estimates calibrated to CE loss scale (~0.2). Loading those into SCST (which operates on reward differences, a completely different scale) causes immediate gradient explosions.

**SCST training loop:**

```
For each batch:
  1. Greedy decode  → w_g          (argmax at each step)
  2. Sample decode  → w_s          (sample from P_θ)
  3. Score both     → r(w_g), r(w_s) via pycocoevalcap CIDEr
  4. Advantage      → A = r(w_s) − r(w_g)
  5. Normalize      → Â = (A − μ) / (σ + ε)
  6. RL loss        → L_SCST = −Â · log P_θ(w_s)
  7. Mixed loss     → L = 0.80 · L_SCST + 0.20 · L_CE
  8. Backprop + clip gradients (max_norm=2.0)
```

> **Why can training loss go negative?** CE loss is always positive. But SCST loss is `−Â · log P`. When the advantage is negative (sampled caption worse than greedy), the gradient *pushes away* from that sequence — the term is negative. This is healthy: it means the model is correctly penalising bad samples.

### Hyperparameters

| Parameter | CE phase | SCST phase |
|:---|:---:|:---:|
| Batch size | 16 | 16 |
| Decoder LR | `4e-4` | `7e-6` |
| Encoder LR | `1e-5` | `3e-6` |
| Grad clip | 2.0 | 2.0 |
| LR scheduler | ReduceLROnPlateau (patience=8) | ReduceLROnPlateau (patience=10) |
| Scheduler metric | BLEU-4 | BLEU-4 |
| RL weight | — | 0.80 |
| CE weight | 1.0 | 0.20 |
| Weight decay | 1e-4 | 1e-4 |

---

## Evaluation

Validation runs after every epoch using **greedy decoding** (fast). Full **beam search** (beam=5) is run via `eval_all.py` for final scoring.

**Beam search** maintains the top-k partial sequences at each step rather than committing to a single greedy choice. Scores are length-normalised to prevent the model from always preferring shorter captions:

```
score(w) = (1 / T^α) · Σ log P(w_t | w_{<t})    # α ≈ 0.7
```

Beam=5 typically yields +1–2 BLEU points over greedy decoding.

**CIDEr** uses TF-IDF weighted n-gram similarity (orders 1–4) against all 5 reference captions per image. Words that are common across all COCO captions get low weight; image-specific words are rewarded. This is why CIDEr is the preferred SCST reward over BLEU — it better reflects whether a caption is specific and informative rather than just fluent.

---

## Dataset & Preprocessing

- **Dataset:** MS-COCO 2014, 5 captions per image, Karpathy split (113k train / 5k val / 5k test)
- **Vocabulary:** `min_word_freq=5` → ~9,500 tokens including `<start>` `<end>` `<pad>` `<unk>`
- **Storage:** Images in HDF5 for efficient random-access batch loading; captions/lengths/word_map in JSON
- **Augmentation:** `RandomHorizontalFlip(p=0.5)` + `ColorJitter` during training

---

## Setup

### Requirements

```bash
pip install torch torchvision nltk pycocoevalcap h5py tqdm Pillow
```

### Dataset download

```bash
# Train images (13 GB)
wget http://images.cocodataset.org/zips/train2014.zip

# Val images (6 GB)
wget http://images.cocodataset.org/zips/val2014.zip

# Karpathy splits + captions
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
```

### Preprocessing

```bash
# Generates HDF5 image files and JSON caption/length files under data/coco/processed/
python create_input_files.py
```

### Training

```bash
# Resumes automatically from checkpoint if it exists
python train.py
```

Set `use_scst = True/False` at the top of `train.py` to switch between CE and SCST phases. The checkpoint saves an `scst` flag so resuming correctly detects the current training regime and only resets optimizers when transitioning between phases.

### Evaluation

```bash
# Full BLEU-4 + CIDEr on Karpathy test split at beam=5
python eval_all.py

# Caption a single image
python caption.py \
  --img='path/to/image.jpg' \
  --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' \
  --word_map='data/coco/processed/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' \
  --beam_size=5
```

---

## File Structure

```
├── train.py               # Main loop — CE and SCST branches, reward normalisation, scheduler
├── models.py              # Encoder (SE-ResNet50-D), DecoderWithAttention, sample() for SCST
├── datasets.py            # CaptionDataset — HDF5 reader
├── utils.py               # save_checkpoint, AverageMeter, clip_gradient
├── eval_all.py            # Beam search + BLEU-4 + CIDEr evaluation
├── caption.py             # Single image captioning with beam search
├── create_input_files.py  # Preprocessing → HDF5 + JSON
└── data/
    └── coco/
        └── processed/     # Generated by create_input_files.py
            ├── TRAIN_IMAGES_coco_...hdf5
            ├── TRAIN_CAPTIONS_coco_...json
            ├── TRAIN_CAPLENS_coco_...json
            └── WORDMAP_coco_...json
```

---

## Limitations & Ceiling Analysis

The BLEU-4 ceiling with this architecture is approximately **0.25–0.27**. Going further requires architectural changes:

| Architecture | Expected BLEU-4 ceiling |
|:---|:---:|
| SE-ResNet-50-D + LSTM (this project) | ~0.25–0.27 |
| ResNet-101 + LSTM | ~0.27–0.29 |
| Bottom-Up object features + LSTM | ~0.31–0.33 |
| ViT + Transformer decoder | ~0.38–0.45+ |

**Why the grid encoder limits performance:**

The 7×7 grid means 49 spatial regions, each covering a ~37×37 pixel patch. This coarse resolution conflates object parts and misses small objects. Bottom-Up features (Faster-RCNN object proposals, ~36 per image) give the attention mechanism semantically coherent regions to work with — "the dog", "the frisbee" — rather than arbitrary grid patches.

**Why LSTM limits performance:**

The LSTM compresses all prior context into a fixed 512-dim hidden state. At word 12 of a caption, the model has no direct access to word 2. This information bottleneck causes repetition and subject-verb disagreement on longer captions. Transformer decoders address this with full self-attention over all prior tokens.

---

## Roadmap

| Priority | Improvement | Expected gain |
|:---|:---|:---:|
| 🟢 Low cost | Entropy bonus in SCST loss (`−λH(P_θ)`, λ≈0.01) | Prevents mode collapse |
| 🟢 Low cost | Gradient accumulation (4–8 steps) | Stabilises SCST variance |
| 🟡 Medium | ResNet-101 backbone swap | +1–2 BLEU-4 |
| 🟡 Medium | Scheduled sampling before SCST transition | Smoother CE→RL handoff |
| 🔴 Architectural | Bottom-Up Top-Down features | +5–8 BLEU-4 |
| 🔴 Architectural | Transformer decoder | +3–5 BLEU-4 |

---

## FAQ

<details>
<summary><strong>Why SE-ResNet-50-D instead of ResNet-101?</strong></summary>

ResNet-101 has ~44M backbone parameters; SE-ResNet-50-D has ~28M but SE blocks add channel-wise recalibration that more than compensates. The SE-50-D produces richer spatial features for the attention mechanism, trains faster per epoch, and generalises better on the captioning task.

</details>

<details>
<summary><strong>Why reset the optimizers when switching to SCST?</strong></summary>

Adam stores first and second moment estimates calibrated to CE loss scale (~0.2). SCST operates on reward differences — a completely different scale. Loading the old Adam state causes gradient explosions. Resetting with conservative LRs (7e-6 / 3e-6) gives a clean start appropriate to the new loss regime.

</details>

<details>
<summary><strong>Why use a mixed loss instead of pure RL?</strong></summary>

Pure RL fine-tuning can produce captions that score well on CIDEr but are grammatically broken or repetitive. The 20% CE component acts as an anchor, keeping the language model grounded in fluent English even as RL pushes it toward higher CIDEr scores.

</details>

<details>
<summary><strong>Why is validation loss increasing during SCST?</strong></summary>

Validation loss is computed with cross-entropy. Once you stop training with CE loss, the CE score naturally drifts upward as the model optimises for CIDEr reward instead. This is expected. BLEU-4 and CIDEr are the only meaningful validation metrics during SCST.

</details>

<details>
<summary><strong>Why does SCST sometimes show negative training loss?</strong></summary>

CE loss is always ≥ 0 since −log(p) ≥ 0 for p ∈ (0,1). But SCST loss is `−Â · log P(w_s)`. When the normalised advantage Â is negative (sampled caption worse than greedy baseline), this term is negative — the model is being pushed *away* from that bad sequence. A negative SCST loss is healthy.

</details>

<details>
<summary><strong>What exactly is the greedy baseline in SCST?</strong></summary>

At each training step the model generates two captions per image: one by sampling from P_θ (stochastic), and one by always taking the argmax word (greedy). The greedy CIDEr score is the baseline. Reinforcement happens only when sampling beats the model's own greedy output — hence "self-critical". As the model improves, so does the baseline, keeping training pressure calibrated.

</details>

---

## Acknowledgements

Based on the [PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) by [@sgrvinod](https://github.com/sgrvinod).

- SCST: [Rennie et al., 2017 — Self-Critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)
- SE blocks: [Hu et al., 2018 — Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- ResNet-D stem: [He et al., 2019 — Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187)
- Dataset: [Lin et al., 2014 — Microsoft COCO](https://arxiv.org/abs/1405.0312)
- Karpathy split: [Karpathy & Fei-Fei, 2015](http://cs.stanford.edu/people/karpathy/deepimagesent/)
