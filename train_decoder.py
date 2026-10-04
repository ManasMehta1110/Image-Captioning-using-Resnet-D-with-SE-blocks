# train_decoder.py
# Trains the attention-LSTM decoder on cached, frozen encoder features.
#   --loss ce     : cross-entropy (teacher forcing), one step per caption
#   --loss scst   : pure SCST (CIDEr-D reward, greedy baseline)
#   --loss mixed  : 0.8 * SCST + 0.2 * CE  (the paper's stabilizer)
# SCST follows the paper's method: greedy-decoding baseline, normalized
# advantage (--no_norm_adv to ablate), fresh Adam at the CE->RL switch
# (--keep_optimizer to ablate). Model selection uses greedy CIDEr on the
# Karpathy VAL split; the test split is never touched here.
import argparse
import glob
import json
import os
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cider_d import CiderD
from evaluate_coco import BASE_DIR, caption_stats
from models import DecoderWithAttention
from utils import clip_gradient

WORD_MAP = os.path.join(BASE_DIR, 'data', 'coco', 'processed', 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
MAX_CAP_LEN = 50  # same caption-length filter as create_input_files


class CaptionHead(nn.Module):
    """Linear projection of frozen encoder features (as Encoder.fc) + attention-LSTM decoder."""

    def __init__(self, feat_dim, vocab_size, d=512, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d) if feat_dim != d else nn.Identity()
        self.decoder = DecoderWithAttention(attention_dim=d, embed_dim=d, decoder_dim=d,
                                            vocab_size=vocab_size, encoder_dim=d, dropout=dropout)


def load_split(feat_dir, split):
    """Returns (path to {split}_feats.npy, meta). Meta lives in feat_dir or its parent
    (timm features share one meta file across encoder subdirs)."""
    path = os.path.join(feat_dir, f'{split}_feats.npy')
    meta_path = os.path.join(feat_dir, f'{split}_meta.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(os.path.normpath(feat_dir)), f'{split}_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    assert len(meta) == np.load(path, mmap_mode='r').shape[0], f'{split}: meta/feature count mismatch'
    return path, meta


class TrainSet(Dataset):
    """One item per image. all_caps=True (CE) returns every caption of the image, so each
    image's features are read once per epoch instead of once per caption; otherwise one
    random caption (the CE term of mixed SCST).
    Memmaps are opened lazily per worker: pickling a memmap would copy the whole array."""

    def __init__(self, parts, word_map, all_caps):
        self.paths = [p for p, _ in parts]
        self.metas = [m for _, m in parts]
        self._feats = None
        self.wm = word_map
        self.all_caps = all_caps
        self.index = [(p, i) for p, meta in enumerate(self.metas) for i in range(len(meta))]

    def __len__(self):
        return len(self.index)

    def encode(self, tokens):
        ids = [self.wm['<start>']] + [self.wm.get(w, self.wm['<unk>']) for w in tokens] + [self.wm['<end>']]
        return ids + [self.wm['<pad>']] * (MAX_CAP_LEN + 2 - len(ids)), len(ids)

    def __getitem__(self, k):
        if self._feats is None:
            self._feats = [np.load(path, mmap_mode='r') for path in self.paths]
        p, i = self.index[k]
        refs = [t for t in self.metas[p][i]['tokens'] if len(t) <= MAX_CAP_LEN]
        chosen = refs if self.all_caps else [random.choice(refs)]
        enc = [self.encode(t) for t in chosen]
        # features stay fp16 until they reach the GPU (half the host->device traffic)
        return torch.from_numpy(np.array(self._feats[p][i])), torch.LongTensor([c for c, _ in enc]), \
            torch.LongTensor([[l] for _, l in enc]), (p, i)


def collate(batch):
    """Returns image features (B, L, D), captions (N, T), lengths (N, 1), and for each
    caption the index of its image in the batch."""
    f, c, l, key = zip(*batch)
    img_idx = torch.cat([torch.full((len(ci),), b, dtype=torch.long) for b, ci in enumerate(c)])
    return torch.stack(f), torch.cat(c), torch.cat(l), img_idx, list(key)


def ids_to_tokens(ids, rev, end):
    out = []
    for w in ids:
        if w == end:
            break
        out.append(rev[w])
    return out


def ce_loss_fn(head, feats, caps, caplens, criterion):
    preds, caps_sorted, decode_lengths, _, _ = head.decoder(feats, caps, caplens)
    targets = caps_sorted[:, 1:preds.size(1) + 1]
    # Token-level mean over the batch in one call: every position past a caption's
    # length has a <pad> target, which criterion ignores (ignore_index).
    return criterion(preds.reshape(-1, preds.size(-1)), targets.reshape(-1))


@torch.no_grad()
def greedy_eval(head, feats, meta, word_map, device, bs=250):
    """Greedy captions + CIDEr / BLEU-4 on a held-out split (pycocoevalcap, Karpathy tokens)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    head.eval()
    rev = {v: k for k, v in word_map.items()}
    caps = []
    for s in range(0, len(meta), bs):
        x = head.proj(torch.from_numpy(np.asarray(feats[s:s + bs], dtype=np.float32)).to(device))
        ids, _ = head.decoder.sample(x, word_map['<start>'], word_map['<end>'], max_len=20, greedy=True)
        caps += [' '.join(ids_to_tokens(r, rev, word_map['<end>'])) for r in ids.tolist()]
    gts = {i: [' '.join(t) for t in m['tokens']] for i, m in enumerate(meta)}
    res = {i: [c if c else 'a'] for i, c in enumerate(caps)}
    cider, _ = Cider().compute_score(gts, res)
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    head.train()
    return {'CIDEr': float(cider), 'BLEU-4': float(bleu[3]), **caption_stats(caps)}


def control_file(out_dir, name):
    """PAUSE / STOP files are honored in the run directory or its parent (runs/ = all runs)."""
    return any(os.path.exists(os.path.join(d, name)) for d in (out_dir, os.path.dirname(os.path.normpath(out_dir))))


def latest_epoch_checkpoint(out_dir):
    eps = []
    for f in glob.glob(os.path.join(out_dir, 'epoch_*.pt')):
        m = re.search(r'epoch_(\d+)\.pt$', f)
        if m:
            eps.append(int(m.group(1)))
    return (max(eps), os.path.join(out_dir, f'epoch_{max(eps)}.pt')) if eps else (0, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feat_dir', required=True, help='dir with {split}_feats.npy (and meta json here or in parent)')
    p.add_argument('--train_splits', nargs='+', default=['train', 'restval'])
    p.add_argument('--loss', choices=['ce', 'scst', 'mixed'], required=True)
    p.add_argument('--rl_weight', type=float, default=0.8, help='mixed: weight of SCST term (CE gets 1 - this)')
    p.add_argument('--no_norm_adv', action='store_true')
    p.add_argument('--init', default=None, help='state_dict (.pt) from a previous run to start from')
    p.add_argument('--init_ep87', action='store_true', help='start from the epoch-87 checkpoint decoder')
    p.add_argument('--keep_optimizer', action='store_true', help='restore Adam state from --init instead of resetting')
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--max_steps_per_epoch', type=int, default=0, help='0 = full pass')
    p.add_argument('--grad_clip', type=float, default=2.0)
    p.add_argument('--patience', type=int, default=3, help='early stop after N epochs without val CIDEr gain')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True, help='run directory')
    p.add_argument('--resume', action='store_true', help='continue from the latest epoch_N.pt in --out')
    args = p.parse_args()

    # SCST lr 5e-5 as in Rennie et al. (2017); the paper's original 1e-6 left val CIDEr unchanged in a pilot
    # batch size counts images; CE uses all ~5 captions of each image (16 images ~= 80 captions)
    defaults = {'ce': (4e-4, 20, 16), 'scst': (5e-5, 10, 32), 'mixed': (5e-5, 10, 32)}[args.loss]
    args.lr = args.lr or defaults[0]
    args.epochs = args.epochs or defaults[1]
    args.batch_size = args.batch_size or defaults[2]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda')
    os.makedirs(args.out, exist_ok=True)
    cfg_name = f"config_resume_{time.strftime('%Y%m%d_%H%M%S')}.json" if args.resume else 'config.json'
    with open(os.path.join(args.out, cfg_name), 'w') as f:
        json.dump(vars(args), f, indent=1)

    with open(WORD_MAP) as f:
        word_map = json.load(f)
    rev = {v: k for k, v in word_map.items()}
    start, end = word_map['<start>'], word_map['<end>']

    parts = [load_split(args.feat_dir, s) for s in args.train_splits]
    val_path, val_meta = load_split(args.feat_dir, 'val')
    val_feats = np.load(val_path, mmap_mode='r')
    feat_dim = val_feats.shape[-1]

    head = CaptionHead(feat_dim, len(word_map)).to(device)
    opt_state = None
    if args.init_ep87:
        ck = torch.load(os.path.join(BASE_DIR, 'checkpoint_coco_5_cap_per_img_5_min_word_freq_epoch_87.pth.tar'),
                        map_location=device, weights_only=False)
        assert feat_dim == 512, 'ep87 decoder expects the ep87 encoder features'
        head.decoder.load_state_dict(ck['decoder'].state_dict())
        opt_state = ck['decoder_optimizer'].state_dict()
    elif args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        head.load_state_dict(ck['model'])
        opt_state = ck.get('optimizer')

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.keep_optimizer:
        assert opt_state is not None, '--keep_optimizer needs an init checkpoint with optimizer state'
        optimizer.load_state_dict(opt_state)
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    loader = DataLoader(TrainSet(parts, word_map, all_caps=args.loss == 'ce'), batch_size=args.batch_size, shuffle=True,
                        num_workers=4, collate_fn=collate, drop_last=True, persistent_workers=True)
    criterion = nn.CrossEntropyLoss(ignore_index=word_map['<pad>']).to(device)
    cider = CiderD([m['tokens'] for _, meta in parts for m in meta]) if args.loss != 'ce' else None
    w_rl = {'ce': 0.0, 'scst': 1.0, 'mixed': args.rl_weight}[args.loss]

    log_path = os.path.join(args.out, 'log.jsonl')
    start_epoch = 1
    if args.resume:
        last, path = latest_epoch_checkpoint(args.out)
        assert path, f'--resume: no epoch_N.pt in {args.out}'
        ck = torch.load(path, map_location=device, weights_only=False)
        head.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        # drop log lines for epochs that never got a checkpoint (e.g. killed while saving)
        with open(log_path) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        recs = [r for r in recs if r['epoch'] <= last]
        with open(log_path, 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in recs)
        if 'best' in ck:
            best, since_best = ck['best'], ck['since_best']
            scheduler.load_state_dict(ck['scheduler'])
            random.setstate(ck['rng']['python'])
            np.random.set_state(ck['rng']['numpy'])
            torch.set_rng_state(ck['rng']['torch'].cpu())
            torch.cuda.set_rng_state(ck['rng']['cuda'].cpu())
        else:  # checkpoint written before resume support: rebuild the bookkeeping from the log
            hist = [(r['epoch'], r['val']['CIDEr']) for r in recs if r['epoch'] > 0 or args.init or args.init_ep87]
            best_ep, best = max(hist, key=lambda x: x[1])
            since_best = last - best_ep
            scheduler.best, scheduler.num_bad_epochs = best, since_best
        start_epoch = last + 1
        print(f"[resume] from {path}: best val CIDEr {best:.4f}, {since_best} epoch(s) since best, "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")
    else:
        val0 = greedy_eval(head, val_feats, val_meta, word_map, device)
        print(f"[init] val {json.dumps({k: round(v, 4) for k, v in val0.items()})}")
        with open(log_path, 'a') as f:
            f.write(json.dumps({'epoch': 0, 'val': val0}) + '\n')
        best, since_best = -1.0, 0
        if args.init or args.init_ep87:  # the starting point counts as a candidate
            best = val0['CIDEr']
            torch.save({'model': head.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0,
                        'feat_dim': feat_dim, 'val': val0, 'args': vars(args)}, os.path.join(args.out, 'best.pt'))

    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        t0, n, sums = time.time(), 0, {'loss': 0., 'ce': 0., 'rl': 0., 'r_sample': 0., 'r_greedy': 0.}
        for step, (feats, caps, caplens, img_idx, keys) in enumerate(tqdm(loader, desc=f'ep{epoch}', leave=False)):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            if step % 25 == 0 and control_file(args.out, 'PAUSE'):
                tqdm.write(f"[pause] PAUSE file found at epoch {epoch} step {step}; GPU idle until it is removed")
                t_pause = time.time()
                while control_file(args.out, 'PAUSE'):
                    time.sleep(15)
                t0 += time.time() - t_pause  # paused time does not count toward epoch time
                tqdm.write("[pause] resumed")
            feats = feats.to(device, non_blocking=True).float()
            caps, caplens, img_idx = caps.to(device), caplens.to(device), img_idx.to(device)
            x = head.proj(feats)
            loss, ce, rl = 0., torch.tensor(0.), torch.tensor(0.)

            if w_rl > 0:
                sampled, logp = head.decoder.sample(x, start, end, max_len=20, greedy=False)
                head.decoder.eval()
                with torch.no_grad():
                    greedy, _ = head.decoder.sample(x, start, end, max_len=20, greedy=True)
                head.decoder.train()
                refs = [parts[p_][1][i]['tokens'] for p_, i in keys]
                r_s = cider.batch_scores([ids_to_tokens(s, rev, end) for s in sampled.tolist()], refs)
                r_g = cider.batch_scores([ids_to_tokens(g, rev, end) for g in greedy.tolist()], refs)
                adv = torch.from_numpy(r_s - r_g).to(device)
                if not args.no_norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                rl = -(adv * logp).mean()
                loss = loss + w_rl * rl
                sums['r_sample'] += float(r_s.mean()); sums['r_greedy'] += float(r_g.mean())
            if w_rl < 1:
                ce = ce_loss_fn(head, x[img_idx], caps, caplens, criterion)
                loss = loss + (1 - w_rl) * ce

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                clip_gradient(optimizer, args.grad_clip)
            optimizer.step()
            n += 1
            sums['loss'] += loss.item(); sums['ce'] += float(ce); sums['rl'] += float(rl)

        train = {k: v / max(n, 1) for k, v in sums.items()}
        val = greedy_eval(head, val_feats, val_meta, word_map, device)
        scheduler.step(val['CIDEr'])
        rec = {'epoch': epoch, 'steps': n, 'secs': round(time.time() - t0), 'lr': optimizer.param_groups[0]['lr'],
               'train': train, 'val': val}
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"[ep{epoch}] {rec['secs']}s train {json.dumps({k: round(v, 4) for k, v in train.items()})} "
              f"| val CIDEr {val['CIDEr']:.4f} BLEU-4 {val['BLEU-4']:.4f} uniq {val['unique_caption_rate']:.3f}")

        improved = val['CIDEr'] > best
        best, since_best = (val['CIDEr'], 0) if improved else (best, since_best + 1)
        state = {'model': head.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'feat_dim': feat_dim, 'val': val, 'args': vars(args),
                 'scheduler': scheduler.state_dict(), 'best': best, 'since_best': since_best,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}}
        # write to a temp file first so a kill during saving never leaves a corrupt epoch file
        tmp = os.path.join(args.out, f'epoch_{epoch}.pt.tmp')
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(args.out, f'epoch_{epoch}.pt'))
        if improved:
            torch.save(state, os.path.join(args.out, 'best.pt'))
        elif since_best >= args.patience:
            print(f"Early stop: no val CIDEr gain for {args.patience} epochs")
            break
        if control_file(args.out, 'STOP'):
            print(f"[stop] STOP file found; epoch {epoch} saved. Delete STOP, then rerun with --resume.")
            return
    print(f"Best val CIDEr {best:.4f}")
    with open(os.path.join(args.out, 'DONE'), 'w') as f:  # run_queue.py skips finished runs
        json.dump({'best_val_cider': best, 'last_epoch': epoch}, f)


if __name__ == '__main__':
    main()
