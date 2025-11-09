import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from models import Encoder, TransformerDecoder, CaptionModel
from datasets import CaptionDataset
from utils import *  # AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_checkpoint, etc.
from nltk.translate.bleu_score import corpus_bleu
import json
import os
from tqdm import tqdm

# Data parameters
data_folder = './data/coco/processed'  # path where your processed files were saved
data_name = 'coco_5_cap_per_img_5_min_word_freq'

# Model parameters
d_model = 512  # dimension of word embeddings / model dim
nhead = 8  # attention heads
num_layers = 3  # decoder layers
dim_feedforward = 2048  # feedforward dim
dropout = 0.1  # dropout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 32
workers = 0  # IMPORTANT: set to 0 on Windows to avoid h5py pickling issues
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.0
best_bleu4 = 0.0
print_freq = 100
fine_tune_encoder = False
checkpoint = "C:\\Users\\Manas Mehta\\Desktop\\PROJECTS\\image-captioningv2\\checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"



def generate_square_subsequent_mask(sz):
    """Generates a square subsequent mask for the Transformer decoder."""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def create_padding_mask(captions, pad_idx):
    """Create padding mask from captions (batch, seq_len) -> (batch, seq_len) bool mask."""
    return (captions == pad_idx)

def greedy_decode(encoder, decoder, encoder_out, max_len=50, sos_idx=0, eos_idx=0, pad_idx=0):
    """Greedy decoding for validation hypotheses."""
    batch_size = encoder_out.size(0)
    hypotheses = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(hypotheses.size(1)).to(device)
        tgt_key_padding_mask = create_padding_mask(hypotheses, pad_idx)

        outputs = decoder(hypotheses, encoder_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        if isinstance(outputs, tuple):  # <── add this line
            outputs = outputs[0]

        next_token = outputs[:, -1, :].argmax(-1).unsqueeze(1)
        hypotheses = torch.cat([hypotheses, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == eos_idx)

        if finished.all():
            break

    return hypotheses


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    sos_idx = word_map['<start>']
    eos_idx = word_map['<end>']
    pad_idx = word_map['<pad>']
    vocab_size = len(word_map)

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder(d_model=d_model)
        encoder.fine_tune(fine_tune_encoder)
        decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                     dim_feedforward=dim_feedforward, dropout=dropout)
        model = CaptionModel(encoder, decoder)
        decoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        encoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) \
            if fine_tune_encoder else None

    else:
        checkpoint_data = torch.load(checkpoint, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        epochs_since_improvement = checkpoint_data['epochs_since_improvement']
        best_bleu4 = checkpoint_data['bleu-4']
        model = checkpoint_data['model']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        encoder = model.encoder
        decoder = model.decoder
        encoder_optimizer = checkpoint_data.get('encoder_optimizer', None)
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr)

    # Move to device
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

    # Custom dataloaders - set num_workers=0 on Windows if using h5py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]))
    val_dataset = CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,  # 0 on Windows to avoid h5py pickling issues
        pin_memory=torch.cuda.is_available()
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder and encoder_optimizer is not None:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              sos_idx=sos_idx,
              eos_idx=eos_idx,
              pad_idx=pad_idx)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                model=model,
                                criterion=criterion,
                                word_map=word_map,
                                sos_idx=sos_idx,
                                eos_idx=eos_idx,
                                pad_idx=pad_idx)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, model, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, model, criterion, encoder_optimizer, decoder_optimizer, epoch, sos_idx, eos_idx, pad_idx):
    """
    Performs one epoch's training.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    # Batches (wrapped with tqdm)
    for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')):
        data_time.update(time.time() - start)

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop. (teacher forcing: input caps[:, :-1], target caps[:, 1:])
        input_captions = caps[:, :-1]
        targets = caps[:, 1:]
        seq_len = input_captions.size(1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
        tgt_key_padding_mask = create_padding_mask(input_captions, pad_idx)

        outputs = model(
            imgs,
            input_captions,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
            )


        # Loss (flatten and ignore pad)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), 5)
        losses.update(loss.item(), imgs.size(0) * seq_len)
        top5accs.update(top5, imgs.size(0) * seq_len)
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, model, criterion, word_map, sos_idx, eos_idx, pad_idx):
    """
    Performs one epoch's validation.
    :return: BLEU-4 score
    """
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(tqdm(val_loader, desc='Validation')):

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop. for loss
            input_captions = caps[:, :-1]
            targets = caps[:, 1:]
            seq_len = input_captions.size(1)
            tgt_mask = generate_square_subsequent_mask(input_captions.size(1)).to(input_captions.device)

            tgt_key_padding_mask = create_padding_mask(input_captions, pad_idx)

            outputs = model(imgs, input_captions, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))

            # Keep track of metrics
            losses.update(loss.item(), imgs.size(0) * seq_len)
            top5 = accuracy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), 5)
            top5accs.update(top5, imgs.size(0) * seq_len)
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            encoder_out = model.encoder(imgs)
            hypotheses_batch = greedy_decode(model.encoder, model.decoder, encoder_out, max_len=50,
                                             sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx)
            hypotheses.extend(hypotheses_batch.tolist())

            # References
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
