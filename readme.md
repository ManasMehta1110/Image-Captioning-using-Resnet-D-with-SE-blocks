<div align="center">

# 🖼️ Image Captioning — SE-ResNet-D + SCST

**Attention-based image captioning on MS-COCO, significantly extending Show, Attend and Tell with stabilized RL.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![COCO](https://img.shields.io/badge/Dataset-MS--COCO-00AABB?style=flat-square)](https://cocodataset.org)
[![Split](https://img.shields.io/badge/Split-Karpathy-6B7280?style=flat-square)](http://cs.stanford.edu/people/karpathy/deepimagesent/)

</div>

---

## Final Results

| Model | Encoder | Training | BLEU-4 | CIDEr |
|:---|:---|:---:|:---:|:---:|
| [Deep VS (Karpathy, 2015)](https://arxiv.org/abs/1412.2306) | VGG-16 | CE only | 0.2250 | — |
| [m-RNN (Mao et al., 2014)](https://arxiv.org/abs/1412.6622) | VGG-16 | CE only | 0.2260 | — |
| **This project (Single Model)** | **SE-ResNet-50-D** | **80/20 Mixed SCST** | **0.2264** | **0.6210** |
| **This project (SOTA Ensemble)** | **SE-ResNet-50-D** | **Polyak Snapshot** | **0.2301** | **0.6236** |
| [Bottom-Up Top-Down (2018)](https://arxiv.org/abs/1707.07998) | Faster-RCNN | CE → SCST | 0.363 | 1.10 |

> **Note:** Our model outperforms foundational baselines (Karpathy/Mao) using a parameter-efficient 512-dim architecture. Measured on Karpathy test split with **Beam size 7** and **4-gram blocking**.

---

## What's Different From the Baseline

### 1. SE-ResNet-50-D Encoder

**ResNet-D stem** replaces the single aggressive 7×7 convolution with three stacked 3×3 convolutions, preserving finer spatial detail for the attention mechanism.

**Squeeze-and-Excitation (SE) blocks** recalibrate channel-wise feature responses:
```
z_c = (1/H×W) Σ u_c(i,j)          # Squeeze: global average pool
s   = σ(W₂ · δ(W₁ · z))           # Excitation: bottleneck MLP → sigmoid
ũ_c = s_c · u_c                    # Scale: channel-wise multiply
```

### 2. SCST with Real CIDEr Reward
We use Self-Critical Sequence Training (REINFORCE) to directly optimize the evaluation metric. The greedy output serves as a baseline — the model is reinforced only when a sample beats its own argmax:
```
L_SCST = −E_{w^s ~ P_θ}[ (r(w^s) − r(w^g)) · log P_θ(w^s) ]
```

### 3. Mixed 80/20 Lost Stabilizer
To prevent "Reward Hacking" (where the model loses grammar while chasing CIDEr points), we anchor the training with 20% Cross-Entropy loss:
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
α_{t,k} = exp(e_{t,k}) / Σ exp(e_{t,j})          # softmax over 49 regions
ẑ_t     = Σ α_{t,k} · A_k                         # context vector
β_t     = σ(W_β · h_{t-1})                        # f_beta visual gate
```

---

## Training Pipeline

The project follows a rigorous four-phase training strategy:

| Phase | Epochs | Goal | Decoder LR | Encoder LR |
|:---|:---:|:---|:---:|:---:|
| **CE Pre-train** | 0–21 | Decoder cold-start (Frozen Encoder) | 4e-4 | 0 |
| **CE Fine-tune** | 21–70 | Joint Cross-Entropy optimization | 4e-4 | 1e-5 |
| **SCST Phase 1** | 71–103| **Stabilized SCST** (80/20 Loss) | 1e-6 | 1e-7 |
| **SCST Phase 2** | 103–110| **Final Fine-Annealing** | 5e-7 | 5e-8 |

**SCST training loop:**
1. **Greedy decode** (`argmax`) to establish the self-critical baseline.
2. **Stochastic sample** from `P_θ` for exploration.
3. **Score both** via `pycocoevalcap` CIDEr.
4. **Normalize Advantage** `Â = (r(s) - r(g) - μ) / σ`.
5. **Backpropagate** using the 80/20 Mixed Loss.

---

## Evaluation (Inference Hacks)

Full **beam search** (beam=7) is run via `eval_ensemble.py`. To solve the "Brevity Penalty," we implement:
*   **Alpha = 0.7:** Length normalization mechanism.
*   **Min-Len = 5:** Blocks `<end>` tokens for the first 5 steps.
*   **4-Gram Blocking:** Prevents repetitive loops common in RL models.
*   **Ensembling:** A Polyak-Averaged snapshot of Epochs [95, 99, 103] with weights [0.1, 0.4, 0.5].

---

## Setup & Execution

### Requirements
```bash
pip install torch torchvision nltk pycocoevalcap h5py tqdm Pillow
```

### Preprocessing
```bash
# Prepares HDF5 images and WORDMAP.json
python create_input_files.py
```

### Training & Evaluation
```bash
# Start Training
python train.py

# Final SOTA Evaluation (Ensemble)
python eval_ensemble.py

# Caption a single image
python caption.py --img='image.jpg' --model='BEST_checkpoint.pth.tar'
```

---

## Limitations & Ceiling Analysis

| Architecture | Expected BLEU-4 ceiling |
|:---|:---:|
| **SE-ResNet-50 + LSTM (This Project)** | **~0.228 — 0.231** |
| ResNet-101 + LSTM | ~0.270 — 0.290 |
| Transformer + ViT | ~0.400+ |

**The Overfitting Discover:** We observed that training beyond Epoch 107 (Loss < -1.0) leads to reward-hacking. The model memorizes training-set noise to farm CIDEr points, resulting in lower generalization on the test split.

---

## FAQ

<details>
<summary><strong>Why use a mixed loss instead of pure RL?</strong></summary>
Pure RL can produce "CIDEr-rich" but grammatically nonsensical captions. The 20% CE component acts as an anchor, keeping the model grounded in fluent English.
</details>

<details>
<summary><strong>Why is validation loss increasing during SCST?</strong></summary>
Validation loss measures Cross-Entropy. As the model optimizes for CIDEr reward, it naturally drifts away from the CE objective. BLEU-4 and CIDEr are the only metrics que matter in Phase 3/4.
</details>

<details>
<summary><strong>Why reset the optimizers when switching to SCST?</strong></summary>
Adam momentum from Phase 2 is calibrated to CE gradients. Transitioning to RL requires fresh momentum estimates to avoid immediate gradient explosions.
</details>

---

## Acknowledgements
Based on the tutorial by [@sgrvinod](https://github.com/sgrvinod).
- SCST: [Rennie et al., 2017](https://arxiv.org/abs/1612.00563)
- SE Blocks: [Hu et al., 2018](https://arxiv.org/abs/1709.01507)
