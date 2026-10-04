# evaluate_coco.py
# Standard COCO-caption evaluation on the Karpathy test split.
#   - Reads test images straight from val2014 JPEGs (no HDF5 needed)
#   - Uses ALL reference captions per image, tokenized with the PTB tokenizer
#   - Reports BLEU-1..4, METEOR, ROUGE-L, CIDEr (+ SPICE with --spice)
#   - Reports caption-quality stats used to quantify reward hacking
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)  # checkpoints pickle models.Encoder / models.DecoderWithAttention

# Words an RL-trained captioner tends to leave dangling at the end of a caption
DANGLING = {'a', 'an', 'the', 'of', 'with', 'and', 'on', 'in', 'at', 'to', 'for', 'is', 'are', 'next', 'its'}


def add_java_to_path(java_home):
    if java_home is None:
        found = glob.glob(os.path.join(os.path.dirname(BASE_DIR), 'tools', 'jdk-*', 'bin'))
        java_home = os.path.dirname(found[0]) if found else None
    if java_home:
        os.environ['PATH'] = os.path.join(java_home, 'bin') + os.pathsep + os.environ['PATH']


class TestImages(Dataset):
    def __init__(self, entries, image_root):
        self.entries = entries
        self.image_root = image_root
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(os.path.join(self.image_root, e['filepath'], e['filename'])).convert('RGB')
        # Same preprocessing as utils.create_input_files: LANCZOS resize to 256x256, uint8
        img = np.array(img.resize((256, 256), Image.LANCZOS)).transpose(2, 0, 1)
        img = torch.FloatTensor(img / 255.)
        return self.normalize(img), i


def banned_by_ngram_block(seq, n):
    """Tokens that would complete an n-gram already present in seq."""
    if n <= 0 or len(seq) < n:
        return set()
    prefix = tuple(seq[-(n - 1):])
    banned = set()
    for j in range(len(seq) - n + 1):
        if tuple(seq[j:j + n - 1]) == prefix:
            banned.add(seq[j + n - 1])
    return banned


@torch.no_grad()
def beam_search(encoder, decoder, image, word_map, beam_size, alpha, min_len, block_ngram, max_steps, device):
    encoder_out = encoder(image.unsqueeze(0).to(device))  # (1, 49, 512)
    return beam_search_from_features(decoder, encoder_out, word_map, beam_size, alpha, min_len,
                                     block_ngram, max_steps, device)


@torch.no_grad()
def beam_search_from_features(decoder, encoder_out, word_map, beam_size, alpha, min_len, block_ngram, max_steps, device):
    """encoder_out: (1, num_pixels, encoder_dim) for a single image."""
    start, end = word_map['<start>'], word_map['<end>']
    vocab_size = len(word_map)

    k = beam_size
    encoder_out = encoder_out.expand(k, encoder_out.size(1), encoder_out.size(2))

    prev_words = torch.full((k, 1), start, dtype=torch.long, device=device)
    seqs = prev_words
    top_k_scores = torch.zeros(k, 1, device=device)
    complete_seqs, complete_scores = [], []

    h, c = decoder.init_hidden_state(encoder_out)

    for step in range(1, max_steps + 1):
        emb = decoder.embedding(prev_words).squeeze(1)
        context, _ = decoder.attention(encoder_out, h)
        context = decoder.sigmoid(decoder.f_beta(h)) * context
        h, c = decoder.decode_step(torch.cat([emb, context], dim=1), (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)

        # step counts generated words so far + this one; block <end> until min_len words exist
        if step <= min_len:
            scores[:, end] = float('-inf')
        if block_ngram:
            for i in range(seqs.size(0)):
                banned = banned_by_ngram_block(seqs[i, 1:].tolist(), block_ngram)
                if banned:
                    scores[i, list(banned)] = float('-inf')

        scores = top_k_scores.expand_as(scores) + scores
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_inds = top_k_words // vocab_size
        next_inds = top_k_words % vocab_size
        seqs = torch.cat([seqs[prev_inds], next_inds.unsqueeze(1)], dim=1)

        incomplete = [i for i, w in enumerate(next_inds.tolist()) if w != end]
        complete = [i for i in range(len(next_inds)) if i not in incomplete]
        for i in complete:
            n_words = seqs.size(1) - 2  # minus <start> and <end>
            complete_seqs.append(seqs[i].tolist())
            complete_scores.append(top_k_scores[i].item() / (max(n_words, 1) ** alpha))

        k -= len(complete)
        if k == 0:
            break
        seqs = seqs[incomplete]
        h = h[prev_inds[incomplete]]
        c = c[prev_inds[incomplete]]
        encoder_out = encoder_out[prev_inds[incomplete]]
        top_k_scores = top_k_scores[incomplete].unsqueeze(1)
        prev_words = next_inds[incomplete].unsqueeze(1)

    if not complete_seqs:  # hit max_steps without any beam finishing
        n_words = seqs.size(1) - 1
        for i in range(seqs.size(0)):
            complete_seqs.append(seqs[i].tolist())
            complete_scores.append(top_k_scores[i].item() / (n_words ** alpha))

    best = complete_seqs[int(np.argmax(complete_scores))]
    return [w for w in best if w not in {start, end, word_map['<pad>']}]


def caption_stats(captions):
    """Caption-quality statistics used to quantify reward hacking."""
    toks = [c.split() for c in captions]
    lengths = [len(t) for t in toks]
    unigrams = Counter(w for t in toks for w in t)
    bigrams = Counter(tuple(t[i:i + 2]) for t in toks for i in range(len(t) - 1))
    return {
        'avg_length': float(np.mean(lengths)),
        'vocab_used': len(unigrams),
        'distinct_1': len(unigrams) / max(sum(unigrams.values()), 1),
        'distinct_2': len(bigrams) / max(sum(bigrams.values()), 1),
        # fraction of captions containing any word more than once, excluding function words
        'repeat_word_rate': float(np.mean([
            any(v > 1 for w, v in Counter(t).items() if w not in DANGLING) for t in toks])),
        # fraction of captions ending on a function word ("... sitting on a table with a")
        'dangling_end_rate': float(np.mean([bool(t) and t[-1] in DANGLING for t in toks])),
        'unique_caption_rate': len(set(captions)) / len(captions),
    }


def score(gts_raw, res_raw, use_spice):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    tok = PTBTokenizer()
    gts = tok.tokenize({k: [{'caption': c} for c in v] for k, v in gts_raw.items()})
    res = tok.tokenize({k: [{'caption': c}] for k, c in res_raw.items()})

    scorers = [(Bleu(4), ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']),
               (Meteor(), 'METEOR'), (Rouge(), 'ROUGE-L'), (Cider(), 'CIDEr')]
    if use_spice:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(), 'SPICE'))

    out = {}
    for scorer, name in scorers:
        s, _ = scorer.compute_score(gts, res)
        if isinstance(name, list):
            out.update({n: float(v) for n, v in zip(name, s)})
        else:
            out[name] = float(s)
        print(f"  {name}: {s}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=os.path.join(
        BASE_DIR, 'checkpoint_coco_5_cap_per_img_5_min_word_freq_epoch_87.pth.tar'))
    p.add_argument('--karpathy_json', default=os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json'))
    p.add_argument('--image_root', default=os.path.join(BASE_DIR, 'data', 'coco', 'images'))
    p.add_argument('--word_map', default=os.path.join(
        BASE_DIR, 'data', 'coco', 'processed', 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'))
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--beam', type=int, default=7)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--min_len', type=int, default=5)
    p.add_argument('--block_ngram', type=int, default=4, help='0 disables n-gram blocking')
    p.add_argument('--max_steps', type=int, default=50)
    p.add_argument('--limit', type=int, default=0, help='evaluate only the first N images (smoke test)')
    p.add_argument('--spice', action='store_true')
    p.add_argument('--java_home', default=None)
    p.add_argument('--tag', default=None, help='name for the output file in results/')
    args = p.parse_args()

    add_java_to_path(args.java_home)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.word_map) as f:
        word_map = json.load(f)
    rev = {v: k for k, v in word_map.items()}

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder, decoder = ckpt['encoder'].to(device).eval(), ckpt['decoder'].to(device).eval()
    assert decoder.fc.out_features == len(word_map), 'word map does not match checkpoint vocab'
    print(f"Checkpoint epoch {ckpt['epoch']} | device {device}")

    with open(args.karpathy_json, encoding='utf-8') as f:
        entries = [e for e in json.load(f)['images'] if e['split'] == args.split]
    if args.limit:
        entries = entries[:args.limit]

    loader = DataLoader(TestImages(entries, args.image_root), batch_size=1, shuffle=False, num_workers=4)

    gts, res, records = {}, {}, []
    t0 = time.time()
    for image, idx in tqdm(loader, desc=f'beam={args.beam}'):
        e = entries[idx.item()]
        words = beam_search(encoder, decoder, image[0], word_map, args.beam, args.alpha,
                            args.min_len, args.block_ngram, args.max_steps, device)
        caption = ' '.join(rev[w] for w in words)
        img_id = e['cocoid']
        gts[img_id] = [s['raw'] for s in e['sentences']]
        res[img_id] = caption
        records.append({'image_id': img_id, 'file': e['filename'], 'caption': caption})
    print(f"Decoded {len(records)} images in {time.time() - t0:.0f}s")

    print('Scoring...')
    metrics = score(gts, res, args.spice)
    stats = caption_stats([r['caption'] for r in records])
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    tag = args.tag or f"ep{ckpt['epoch']}_{args.split}_b{args.beam}_a{args.alpha}_ml{args.min_len}_ng{args.block_ngram}"
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    out = os.path.join(BASE_DIR, 'results', f'eval_{tag}.json')
    with open(out, 'w') as f:
        json.dump({'config': vars(args), 'checkpoint_epoch': ckpt['epoch'], 'n_images': len(records),
                   'metrics': metrics, 'caption_stats': stats, 'captions': records}, f, indent=1)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
