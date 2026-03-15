# Image Captioning using ResNet-D with SE Blocks

A PyTorch implementation of attention-based image captioning, significantly extending the classic [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) architecture with three key upgrades: a stronger SE-ResNet-D encoder, Self-Critical Sequence Training (SCST) with CIDEr reward, and a mixed CE + RL loss for stable fine-tuning.

---

## Results

| Model | Encoder | Training | BLEU-4 | CIDEr |
|:---:|:---:|:---:|:---:|:---:|
| Show, Attend and Tell (original) | ResNet-101 | CE only | ~24.3 | - |
| **This project** | **SE-ResNet-50-D** | **CE → SCST** | **~21.8** | **~0.60** |

> Results measured on MSCOCO Karpathy test split at beam size 5. Training was run for 76 epochs total — 70 epochs of cross-entropy followed by SCST fine-tuning. Scores are expected to improve further with continued SCST training.

---

## What's Different From the Baseline

### 1. SE-ResNet-50-D Encoder (vs. ResNet-101)

The original tutorial uses a standard ResNet-101 encoder. This project replaces it with a **SE-ResNet-50-D** — a ResNet-50 with two architectural improvements:

**ResNet-D stem**: Instead of a single aggressive 7×7 convolution, the stem uses three stacked 3×3 convolutions. This preserves more fine-grained spatial detail in early layers, which is important for attention — the model needs rich spatial features to attend to.

**Squeeze-and-Excitation (SE) blocks**: After every bottleneck block, an SE block recalibrates channel-wise feature responses. It globally pools each feature map, passes it through a small bottleneck MLP, and produces per-channel weights via sigmoid. This lets the network learn *which feature maps matter* for a given image, acting as a learned attention mechanism at the feature level — separate from and complementary to the spatial attention in the decoder.

The result is a more expressive encoder with fewer parameters than ResNet-101, which trains faster and generalises better on the caption generation task.

### 2. Self-Critical Sequence Training (SCST) with CIDEr Reward

Standard cross-entropy training suffers from **exposure bias** — during training the decoder sees ground-truth words, but at inference it must rely on its own previous predictions. This mismatch degrades output quality.

SCST addresses this directly using Reinforcement Learning. The model generates a caption by sampling words from its own distribution (no teacher forcing), scores it using the CIDEr metric against the ground truth references, and uses that score as a reward signal. The key insight from the SCST paper is using the model's own **greedy decode** as the baseline:

$$\mathcal{L}_{SCST} = -\mathbb{E}_{w^s \sim p_\theta}\left[r(w^s) - r(w^g)\right]$$

Where $r(w^s)$ is the CIDEr score of the sampled caption and $r(w^g)$ is the CIDEr score of the greedy caption. If the sampled caption beats greedy, the model is reinforced. If it does worse, it is penalised. This directly optimises the evaluation metric rather than a proxy loss.

To stabilise training, rewards are normalised per batch:

$$\hat{r} = \frac{r - \mu_r}{\sigma_r + \epsilon}$$

This prevents the gradient scale from varying wildly between batches, which is especially important with small batch sizes.

### 3. Mixed CE + RL Loss

Switching abruptly from cross-entropy to pure RL can destabilise training — the model can forget grammatical structure while chasing reward. This project uses a **mixed loss** during SCST fine-tuning:

$$\mathcal{L} = 0.80 \cdot \mathcal{L}_{SCST} + 0.20 \cdot \mathcal{L}_{CE}$$

The 20% CE component acts as an anchor, preventing the model from drifting too far from fluent language even as the RL component pushes it toward higher CIDEr scores. This produces more readable captions than pure RL while still capturing the metric optimisation benefits.

---

## Overview

### Encoder

The encoder is a **SE-ResNet-50-D** pretrained on ImageNet. After the final residual stage, an `AdaptiveAvgPool2d` resizes the feature map to 7×7, producing 49 spatial locations. A linear projection then maps each location from 2048 dimensions to the model's internal dimension (512). This gives the decoder 49 feature vectors to attend over.

Only layers 2–4 of the ResNet are fine-tuned during training. Layer 1 captures low-level features (edges, textures) that are universal and don't benefit from task-specific tuning.

### Attention

Soft attention is used, identical in structure to the original paper. At each decoder timestep, the attention network takes the encoder features and the decoder's current hidden state, and produces a weight for each of the 49 spatial locations. The weighted sum of encoder features is the context vector fed to the LSTM.

A sigmoid gate (the `f_beta` layer from the paper) modulates the context vector using the previous hidden state, helping the attention focus more sharply on objects.

### Decoder

A single-layer LSTM with hidden size 512. At each timestep it receives the embedding of the previously generated word concatenated with the gated context vector from attention. Its output is projected to vocabulary size to produce word scores.

During **cross-entropy training**, the decoder is teacher-forced — it receives ground-truth words as input at every step.

During **SCST training**, teacher forcing is disabled. The decoder generates captions entirely from its own predictions via sampling (for the RL update) and greedy argmax (for the baseline).

---

## Training Procedure

Training is done in two phases:

**Phase 1 — Cross-Entropy (Epochs 0–70)**

Standard teacher-forced training with cross-entropy loss. The encoder is fine-tuned from epoch 0. A `ReduceLROnPlateau` scheduler reduces LRs on BLEU-4 plateau with patience of 8 epochs. This phase brings BLEU-4 to approximately 0.188.

**Phase 2 — SCST Fine-tuning (Epoch 71+)**

The optimizers are reset with much lower learning rates (decoder: `7e-6`, encoder: `3e-6`) to avoid disturbing the well-trained weights. The loss switches to the mixed CE + RL objective. The scheduler patience is increased to 10 epochs to account for the noisier reward signal.

The checkpoint saves an `scst` flag so that if training is resumed, the code correctly detects whether the saved checkpoint was CE or SCST trained, and resets optimizers only when transitioning between regimes.

---

## Setup

### Requirements

```
torch
torchvision
nltk
pycocoevalcap
h5py
tqdm
Pillow
```

Install with:

```bash
pip install torch torchvision nltk pycocoevalcap h5py tqdm Pillow
```

### Dataset

Download the MSCOCO 2014 dataset:
- [Train images (13GB)](http://images.cocodataset.org/zips/train2014.zip)
- [Val images (6GB)](http://images.cocodataset.org/zips/val2014.zip)
- [Karpathy splits + captions](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

### Preprocessing

Point `create_input_files.py` to your Karpathy JSON and image folders, then run:

```bash
python create_input_files.py
```

This creates HDF5 image files and JSON caption/length files under `data/coco/processed/`.

---

## Training

```bash
python train.py
```

Training automatically resumes from `checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar` if it exists. The best model is saved separately as `BEST_checkpoint_...`.

Key hyperparameters at the top of `train.py`:

| Parameter | CE Phase | SCST Phase |
|:---|:---:|:---:|
| Batch size | 16 | 16 |
| Decoder LR | `4e-4` | `7e-6` |
| Encoder LR | `1e-5` | `3e-6` |
| Grad clip | 2.0 | 2.0 |
| Scheduler patience | 8 | 10 |
| Max length (SCST sample) | — | 20 |
| RL weight | — | 0.80 |
| CE weight | — | 0.20 |

To switch between CE and SCST training, set `use_scst = True/False` at the top of `train.py`. The transition is handled automatically.

---

## Evaluation

```bash
python eval_all.py
```

Evaluates the best checkpoint on the MSCOCO Karpathy test split at beam size 5, reporting BLEU-4 and CIDEr.

To caption a single image:

```bash
python caption.py --img='path/to/image.jpg' \
                  --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' \
                  --word_map='data/coco/processed/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' \
                  --beam_size=5
```

---

## File Structure

```
├── train.py              # Training loop (CE + SCST)
├── models.py             # SE-ResNet-D encoder + attention decoder
├── datasets.py           # CaptionDataset (HDF5 reader)
├── utils.py              # save_checkpoint, AverageMeter, etc.
├── eval_all.py           # BLEU-4 + CIDEr evaluation
├── caption.py            # Single image captioning with beam search
├── create_input_files.py # Preprocessing script
└── data/
    └── coco/
        └── processed/    # HDF5 + JSON files generated by preprocessing
```

---

## FAQs

**Why SE-ResNet-50-D instead of ResNet-101?**

ResNet-101 has ~44M parameters in the backbone. SE-ResNet-50-D has ~28M but with SE blocks adds channel-wise recalibration that more than compensates. In practice the SE-50-D produces richer spatial features for the attention mechanism to work with, and trains faster per epoch.

**Why reset the optimizers when switching to SCST?**

Adam stores momentum and variance estimates for every parameter. These are calibrated to the CE loss scale (~0.2). The SCST loss operates on a completely different scale (reward differences rather than token probabilities). Loading the old Adam state into SCST training causes gradient explosions. Resetting with low LRs gives the optimizer a clean start appropriate to the new loss regime.

**Why use a mixed loss instead of pure RL?**

Pure RL fine-tuning can cause the model to produce high-CIDEr captions that are grammatically broken or repetitive. The 20% CE component keeps the language model grounded during RL training, producing captions that score well on metrics and read naturally.

**What is the greedy baseline in SCST?**

At each training step, the model generates two captions for every image: one by sampling from the predicted distribution (stochastic), and one by always picking the highest-probability word (greedy). The greedy caption's CIDEr score is the baseline. The model is only reinforced when its sampled caption beats its own greedy output — this is what makes the training self-critical.

**Why is val loss increasing during SCST?**

Validation loss is computed with cross-entropy. Once you stop training with CE loss, the CE score will naturally drift upward as the model optimises for CIDEr reward instead. This is expected and not a sign of degradation. BLEU-4 is the only meaningful validation metric during SCST training.

---

## Acknowledgements

Based on the excellent [PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) by [sgrvinod](https://github.com/sgrvinod). The SCST training methodology follows [Rennie et al., 2017](https://arxiv.org/abs/1612.00563). SE blocks are from [Hu et al., 2018](https://arxiv.org/abs/1709.01507). ResNet-D stem design from [He et al., 2019](https://arxiv.org/abs/1812.01187).
