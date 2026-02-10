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

from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm  # ✅ tqdm added

# 🔁 CHANGED IMPORT (Transformer → LSTM)
from models import Encoder, DecoderWithAttention

from datasets import *
from utils import *

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
batch_size = 32
workers = 0  # ✅ Windows-safe (h5py)
encoder_lr = 1e-5
decoder_lr = 4e-4
grad_clip = 5.
best_bleu4 = 0.
print_freq = 100

# Phase flags
load_imagenet_weights = True
fine_tune_encoder = True

checkpoint = None

# -------- CHECKPOINT AUTO-RESUME --------
checkpoint_path = os.path.join(
    BASE_DIR,
    f'checkpoint_{data_name}.pth.tar'
)

if os.path.isfile(checkpoint_path):
    checkpoint = checkpoint_path
    print(f"=> Found checkpoint at {checkpoint_path}, resuming training")
else:
    print("=> No checkpoint found, starting fresh")
# --------------------------------------

log_file = 'train_log.txt'

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)


def main():
    global best_bleu4, epochs_since_improvement, checkpoint
    global start_epoch, fine_tune_encoder, data_name, word_map

    # Load word map
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

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
            lr=decoder_lr
        )

        encoder = Encoder(
            d_model=emb_dim,
            load_imagenet=load_imagenet_weights
        )

        encoder.fine_tune(fine_tune_encoder)

        encoder_optimizer = (
            torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr
            ) if fine_tune_encoder else None
        )

    else:
        checkpoint = torch.load(checkpoint, weights_only=False)

        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']

        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']

        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(True)
            encoder_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr
            )

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=word_map['<pad>']).to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'TRAIN',
            transform=transforms.Compose([normalize])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'VAL',
            transform=transforms.Compose([normalize])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):

        if epochs_since_improvement == 20:
            break

        train_loss, train_top5acc = train(
            train_loader, encoder, decoder,
            criterion, encoder_optimizer, decoder_optimizer, epoch
        )

        val_loss, val_top5acc, recent_bleu4 = validate(
            val_loader, encoder, decoder, criterion
        )

        logging.info(
            f'Epoch {epoch} | '
            f'Train Loss {train_loss.avg:.4f} | '
            f'Val Loss {val_loss.avg:.4f} | '
            f'BLEU-4 {recent_bleu4:.4f}'
        )

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(best_bleu4, recent_bleu4)

        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        save_checkpoint(
            data_name, epoch, epochs_since_improvement,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer,
            recent_bleu4, is_best
        )


def train(train_loader, encoder, decoder, criterion,
          encoder_optimizer, decoder_optimizer, epoch):

    decoder.train()
    encoder.train()

    losses = AverageMeter()
    top5accs = AverageMeter()

    for imgs, caps, caplens in tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False):

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)

        predictions, caps_sorted, decode_lengths, _, sort_ind = decoder(
            imgs, caps, caplens
        )

        targets = caps_sorted[:, 1:]

        loss = 0
        for i, l in enumerate(decode_lengths):
            loss += criterion(
                predictions[i, :l, :],
                targets[i, :l]
            )

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

        top5 = accuracy(
            predictions[:, :max(decode_lengths), :].contiguous().view(-1, predictions.size(-1)),
            targets[:, :max(decode_lengths)].contiguous().view(-1),
            5
        )

        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))

    return losses, top5accs


def validate(val_loader, encoder, decoder, criterion):
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

            predictions, caps_sorted, decode_lengths, _, sort_ind = decoder(
                imgs, caps, caplens
            )

            targets = caps_sorted[:, 1:]

            loss = 0
            for i, l in enumerate(decode_lengths):
                loss += criterion(
                    predictions[i, :l, :],
                    targets[i, :l]
                )

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
                    [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}]
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
    main()
