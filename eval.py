# eval.py
import json
import math
import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from datasets import *
from utils import *
from models import Encoder, DecoderWithAttention
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# =======================
# PATHS & PARAMETERS
# =======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(BASE_DIR, 'data', 'coco', 'processed')
data_name = 'coco_5_cap_per_img_5_min_word_freq'

checkpoint_path = os.path.join(
    BASE_DIR,
    'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
)

word_map_file = os.path.join(
    data_folder,
    'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# =======================
# LOAD CHECKPOINT
# =======================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

encoder = checkpoint['encoder'].to(device)
decoder = checkpoint['decoder'].to(device)

encoder.eval()
decoder.eval()

# =======================
# LOAD WORD MAP
# =======================

with open(word_map_file, 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# =======================
# NORMALIZATION
# =======================

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# =======================
# BEAM SEARCH EVALUATION
# =======================

def evaluate(beam_size):
    """
    Evaluation WITHOUT teacher forcing (true inference with beam search)
    """

    loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder,
            data_name,
            'TEST',
            transform=transforms.Compose([normalize])
        ),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    references = []
    hypotheses = []

    for image, caps, caplens, allcaps in tqdm(
        loader, desc=f"EVALUATING @ BEAM SIZE {beam_size}"
    ):

        image = image.to(device)

        # =======================
        # ENCODE IMAGE
        # =======================

        encoder_out = encoder(image)            # (1, num_pixels, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)

        # =======================
        # INITIALIZE BEAM
        # =======================

        k = beam_size
        prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = prev_words

        top_k_scores = torch.zeros(k, 1).to(device)

        complete_seqs = []
        complete_seqs_scores = []

        # Initialize LSTM state
        h, c = decoder.init_hidden_state(encoder_out)

        step = 1

        while True:

            embeddings = decoder.embedding(prev_words).squeeze(1)

            context, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context

            h, c = decoder.decode_step(
                torch.cat([embeddings, context], dim=1),
                (h, c)
            )

            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)],
                dim=1
            )

            incomplete_inds = [
                i for i, w in enumerate(next_word_inds)
                if w != word_map['<end>']
            ]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)

            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break

            step += 1

        best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[best_idx]

        # =======================
        # REFERENCES
        # =======================

        img_caps = allcaps[0].tolist()
        img_captions = [
            [w for w in c if w not in
             {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            for c in img_caps
        ]
        references.append(img_captions)

        # =======================
        # HYPOTHESIS
        # =======================

        hypotheses.append([
            w for w in seq
            if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}
        ])

    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4


# =======================
# MAIN
# =======================

if __name__ == '__main__':
    beam_size = 5
    bleu4 = evaluate(beam_size)
    print(f"\nBLEU-4 @ beam size {beam_size}: {bleu4:.4f}")
