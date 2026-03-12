# train.py
import time
import json
import os
import logging
import math

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Categorical

from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider   # Real CIDEr reward
from tqdm import tqdm

from models import Encoder, DecoderWithAttention

from datasets import *
from utils import *

# ============================
# === SCST SETTINGS ==========
# ============================
use_scst = True

# Data parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(BASE_DIR, 'data', 'coco', 'processed')
data_name = 'coco_5_cap_per_img_5_min_word_freq'

word_map_file = os.path.join(
    data_folder,
    'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
)

# Model parameters
emb_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 16
workers = 0

# === LR settings ===
# These are used only when starting fresh (no checkpoint) under CE loss.
# When transitioning from a CE checkpoint to SCST, fresh optimizers are
# created with the SCST-safe LRs defined below.
encoder_lr = 1e-5
decoder_lr = 4e-4

# SCST-safe learning rates — used when a CE checkpoint is loaded and
# use_scst=True, because the old optimizer state is discarded.
scst_decoder_lr = 7e-6
scst_encoder_lr = 3e-6

grad_clip = 2.      # Tighter clip for SCST stability (was 5.0)
best_bleu4 = 0.

# Phase flags
load_imagenet_weights = True
fine_tune_encoder = True

checkpoint = None

checkpoint_path = os.path.join(
    BASE_DIR,
    f'checkpoint_{data_name}_epoch_77.pth.tar'
)

if os.path.isfile(checkpoint_path):
    checkpoint = checkpoint_path
    print(f"=> Found checkpoint at {checkpoint_path}, resuming training")
else:
    print("=> No checkpoint found, starting fresh")

log_file = 'train_log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')


def main():
    global best_bleu4, epochs_since_improvement, checkpoint
    global start_epoch, fine_tune_encoder, data_name, word_map, idx2word

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    idx2word = {v: k for k, v in word_map.items()}

    # =====================
    # Initialize models
    # =====================
    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=512,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            encoder_dim=emb_dim,
            dropout=dropout
        )

        decoder_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr if not use_scst else scst_decoder_lr,
            weight_decay=1e-4
        )

        encoder = Encoder(d_model=emb_dim, load_imagenet=load_imagenet_weights)
        encoder.fine_tune(fine_tune_encoder)

        encoder_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=encoder_lr if not use_scst else scst_encoder_lr,
            weight_decay=1e-4
        ) if fine_tune_encoder else None

    else:
        checkpoint_data = torch.load(checkpoint, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        epochs_since_improvement = checkpoint_data['epochs_since_improvement']
        best_bleu4 = checkpoint_data['bleu-4']

        decoder = checkpoint_data['decoder']
        encoder = checkpoint_data['encoder']

        # === KEY FIX ===
        # Detect whether the checkpoint was trained with CE or SCST by checking
        # whether it carries an 'scst' flag.  Old CE checkpoints won't have it,
        # so we default to False and treat the loaded optimizers as stale.
        checkpoint_was_scst = checkpoint_data.get('scst', False)

        if use_scst and not checkpoint_was_scst:
            # Transitioning from CE → SCST:
            # Discard the old Adam momentum state entirely and create fresh
            # optimizers with conservative SCST-safe learning rates.
            print("=> Detected CE→SCST transition: resetting optimizers with SCST LRs")
            decoder_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, decoder.parameters()),
                lr=scst_decoder_lr,
                weight_decay=1e-4
            )
            encoder_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=scst_encoder_lr,
                weight_decay=1e-4
            ) if fine_tune_encoder else None
        else:
            # Same regime as checkpoint — safe to restore optimizer state.
            decoder_optimizer = checkpoint_data['decoder_optimizer']
            encoder_optimizer = checkpoint_data['encoder_optimizer']

            if fine_tune_encoder and encoder_optimizer is None:
                encoder.fine_tune(True)
                encoder_optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, encoder.parameters()),
                    lr=scst_encoder_lr if use_scst else encoder_lr,
                    weight_decay=1e-4
                )

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Looser patience when doing SCST (reward signal is noisier)
    scheduler_patience = 10 if use_scst else 8

    decoder_scheduler = ReduceLROnPlateau(
        decoder_optimizer, mode='max',
        patience=scheduler_patience, factor=0.5, min_lr=1e-7
    )
    if encoder_optimizer is not None:
        encoder_scheduler = ReduceLROnPlateau(
            encoder_optimizer, mode='max',
            patience=scheduler_patience, factor=0.5, min_lr=1e-7
        )

    criterion = nn.CrossEntropyLoss(ignore_index=word_map['<pad>']).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )

    cider_scorer = Cider()

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):

        if epochs_since_improvement == 20:
            break

        train_loss, train_top5acc = train(
            train_loader, encoder, decoder, criterion,
            encoder_optimizer, decoder_optimizer, epoch, word_map, cider_scorer
        )

        val_loss, val_top5acc, recent_bleu4 = validate(val_loader, encoder, decoder, criterion, word_map)

        logging.info(f'Epoch {epoch} | Train Loss {train_loss.avg:.4f} | Val Loss {val_loss.avg:.4f} | BLEU-4 {recent_bleu4:.4f}')

        print(f"Epoch {epoch} | Decoder LR: {decoder_optimizer.param_groups[0]['lr']:.2e}")
        if encoder_optimizer is not None:
            print(f"Epoch {epoch} | Encoder LR: {encoder_optimizer.param_groups[0]['lr']:.2e}")

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(best_bleu4, recent_bleu4)

        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        decoder_scheduler.step(recent_bleu4)
        if encoder_optimizer is not None:
            encoder_scheduler.step(recent_bleu4)

        save_checkpoint(data_name, epoch, epochs_since_improvement,
                        encoder, decoder, encoder_optimizer, decoder_optimizer,
                        recent_bleu4, is_best,
                        scst=use_scst)   # <-- store training regime in checkpoint


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map, cider_scorer):
    decoder.train()
    encoder.train()

    losses = AverageMeter()
    top5accs = AverageMeter()

    start_token = word_map.get('<start>', 0)
    end_token   = word_map.get('<end>', 1)
    pad_token   = word_map.get('<pad>', 2)

    for imgs, caps, caplens in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False):

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)

        if use_scst:
            # ===================== REAL CIDEr SCST =====================
            sampled_ids, log_probs = decoder.sample(imgs, max_len=20, greedy=False)

            with torch.no_grad():
                greedy_ids, _ = decoder.sample(imgs, max_len=20, greedy=True)

            references = {}
            sampled_dict = {}
            greedy_dict = {}

            for i in range(caps.size(0)):
                # Reference (ground truth)
                ref = []
                for w in caps[i].tolist():
                    if w == end_token:
                        break
                    if w not in {start_token, pad_token}:
                        ref.append(idx2word.get(w, '<unk>'))
                references[i] = [' '.join(ref)] if ref else [' ']

                # Sampled
                hyp_s = []
                for w in sampled_ids[i].tolist():
                    if w == end_token:
                        break
                    if w not in {start_token, pad_token}:
                        hyp_s.append(idx2word.get(w, '<unk>'))
                sampled_dict[i] = [' '.join(hyp_s)] if hyp_s else [' ']

                # Greedy baseline
                hyp_g = []
                for w in greedy_ids[i].tolist():
                    if w == end_token:
                        break
                    if w not in {start_token, pad_token}:
                        hyp_g.append(idx2word.get(w, '<unk>'))
                greedy_dict[i] = [' '.join(hyp_g)] if hyp_g else [' ']

            # Get per-sample CIDEr scores
            _, sampled_scores = cider_scorer.compute_score(references, sampled_dict)
            _, greedy_scores  = cider_scorer.compute_score(references, greedy_dict)

            sampled_scores = torch.tensor(sampled_scores, dtype=torch.float, device=device)
            greedy_scores  = torch.tensor(greedy_scores, dtype=torch.float, device=device)

            # Normalise the reward signal to reduce scale sensitivity
            rewards = sampled_scores - greedy_scores
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            loss = -(rewards * log_probs).mean()

        else:
            predictions, caps_sorted, decode_lengths, _, sort_ind = decoder(imgs, caps, caplens)
            targets = caps_sorted[:, 1:]

            loss = 0
            for i, l in enumerate(decode_lengths):
                loss += criterion(predictions[i, :l, :], targets[i, :l])
            loss = loss / sum(decode_lengths)

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        if not use_scst:
            top5 = accuracy(
                predictions[:, :max(decode_lengths), :].contiguous().view(-1, predictions.size(-1)),
                targets[:, :max(decode_lengths)].contiguous().view(-1),
                5
            )
            top5accs.update(top5, sum(decode_lengths))

        losses.update(loss.item(), imgs.size(0))

    return losses, top5accs


def validate(val_loader, encoder, decoder, criterion, word_map):
    decoder.eval()
    encoder.eval()

    losses = AverageMeter()
    top5accs = AverageMeter()

    references = []
    hypotheses = []

    with torch.no_grad():
        for imgs, caps, caplens, allcaps in tqdm(val_loader, desc="Validation", leave=False):

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs = encoder(imgs)

            predictions, caps_sorted, decode_lengths, _, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            loss = 0
            for i, l in enumerate(decode_lengths):
                loss += criterion(predictions[i, :l, :], targets[i, :l])
            loss = loss / sum(decode_lengths)
            losses.update(loss.item(), sum(decode_lengths))

            top5 = accuracy(
                predictions[:, :max(decode_lengths), :].contiguous().view(-1, predictions.size(-1)),
                targets[:, :max(decode_lengths)].contiguous().view(-1),
                5
            )
            top5accs.update(top5, sum(decode_lengths))

            allcaps = allcaps[sort_ind.cpu()]
            for j in range(allcaps.size(0)):
                references.append([
                    [w for w in c if w not in {word_map.get('<start>', -1), word_map.get('<pad>', -1)}]
                    for c in allcaps[j].tolist()
                ])

            _, preds = torch.max(predictions, dim=2)
            preds = preds.tolist()
            preds = [p[:decode_lengths[j]] for j, p in enumerate(preds)]
            hypotheses.extend(preds)

        bleu4 = corpus_bleu(references, hypotheses)

    return losses, top5accs, bleu4


if __name__ == '__main__':
    print("Using device:", device)
    print("Load ImageNet weights:", load_imagenet_weights)
    print("Fine-tune encoder:", fine_tune_encoder)
    print("SCST enabled:", use_scst)
    print("Using REAL CIDEr reward (per-sample, normalised)")
    main()

