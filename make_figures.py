# make_figures.py
# Builds the paper's result figures from saved logs, test results and checkpoints:
#   paper/figs/training_dynamics.pdf  - val CIDEr and dangling-ending rate per SCST epoch
#   paper/figs/qualitative.pdf        - test images with a reference, CE, mixed and pure SCST captions
#   paper/figs/attention_maps.pdf     - per-word attention of the mixed SCST model (greedy decoding)
# Example images are drawn at random with a fixed seed, not hand-picked.
import json
import os
import random
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_coco import BASE_DIR, DANGLING
from train_decoder import WORD_MAP, CaptionHead

FIG_DIR = os.path.join(BASE_DIR, 'paper', 'figs')
SEED = 0

# Categorical slots 1-4 of the reference palette, assigned in fixed order.
# Identity never rests on color alone: each series also has its own marker,
# dash style and a direct label.
SERIES = [
    ('scst_mixed_se_s0', 'Mixed (0.8/0.2)', '#2a78d6', 'o', '-'),
    ('scst_pure_se_s0', 'Pure SCST', '#eb6834', 's', '--'),
    ('scst_mixed_nonorm_se_s0', 'Mixed, no adv. norm.', '#1baf7a', '^', '-.'),
    ('scst_mixed_noreset_se_s0', 'Mixed, no opt. reset', '#eda100', 'D', ':'),
]
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dd'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'pdf.fonttype': 42,
})


def load_log(run):
    with open(os.path.join(BASE_DIR, 'runs', run, 'log.jsonl')) as f:
        return [json.loads(l) for l in f if l.strip()]


def training_dynamics():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True,
                                   gridspec_kw={'height_ratios': [1.25, 1]})
    for run, label, color, marker, dash in SERIES:
        rows = load_log(run)
        ep = [r['epoch'] for r in rows]
        cider = [r['val']['CIDEr'] * 100 for r in rows]
        dangl = [r['val']['dangling_end_rate'] * 100 for r in rows]
        kw = dict(color=color, marker=marker, linestyle=dash, linewidth=1.3, markersize=3.5,
                  markeredgecolor='white', markeredgewidth=0.5, label=label)
        ax1.plot(ep, cider, **kw)
        ax2.plot(ep, dangl, **kw)
    ce = load_log('scst_mixed_se_s0')[0]['val']['CIDEr'] * 100
    ax1.axhline(ce, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    # inside the plot, above the line, where no series passes (all lines are above 91 there)
    ax1.text(5.0, ce + 0.5, 'CE start', va='bottom', fontsize=6.5, color=MUTED)
    ax1.set_ylabel('Val. CIDEr')
    ax2.set_ylabel('Dangling endings (%)')
    ax2.set_xlabel('SCST epoch')
    ax2.set_xticks(range(0, 11))
    ax2.set_ylim(-4, 85)
    # Direct label for the one series that leaves zero; the three others overlap at 0%.
    ax2.annotate('Pure SCST', xy=(10, load_log('scst_pure_se_s0')[-1]['val']['dangling_end_rate'] * 100),
                 xytext=(7.3, 70), fontsize=6.5, color=INK,
                 arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.6))
    ax2.text(10.3, 5, 'mixed variants:\nat most 0.04%', fontsize=6.5, color=MUTED, ha='right', va='bottom')
    for ax in (ax1, ax2):
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    ax1.legend(loc='lower right', frameon=False, ncol=1, handlelength=2.6)
    fig.tight_layout(h_pad=0.6)
    out = os.path.join(FIG_DIR, 'training_dynamics.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def load_results(run):
    path = os.path.join(BASE_DIR, 'results', f'eval_{run}_test_b7_a0.7_ml5_ng4.json')
    with open(path) as f:
        return {c['image_id']: c for c in json.load(f)['captions']}


def load_image(filename):
    folder = 'val2014' if 'val2014' in filename else 'train2014'
    img = Image.open(os.path.join(BASE_DIR, 'data', 'coco', 'images', folder, filename)).convert('RGB')
    return img.resize((256, 256), Image.LANCZOS)


def qualitative(n=3):
    ce, mixed, pure = (load_results(r) for r in ('ce_seresnet50_s0', 'scst_mixed_se_s0', 'scst_pure_se_s0'))
    with open(os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json'), encoding='utf-8') as f:
        refs = {e['cocoid']: e['sentences'][0]['raw'].strip() for e in json.load(f)['images'] if e['split'] == 'test'}
    ids = random.Random(SEED).sample(sorted(ce), n)

    fig, axes = plt.subplots(n, 2, figsize=(3.45, 0.8 * n), gridspec_kw={'width_ratios': [1, 2.7]})
    for row, img_id in enumerate(ids):
        axi, axt = axes[row]
        axi.imshow(load_image(ce[img_id]['file']))
        axi.axis('off')
        axt.axis('off')
        lines = [('GT', refs[img_id].rstrip('.').lower()), ('CE', ce[img_id]['caption']),
                 ('Mixed', mixed[img_id]['caption']), ('Pure', pure[img_id]['caption'])]
        y = 0.98
        for tag, cap in lines:
            wrapped = textwrap.fill(cap, 48)
            axt.text(0.0, y, f'{tag}:', fontsize=6, fontweight='bold', va='top', transform=axt.transAxes)
            axt.text(0.2, y, wrapped, fontsize=6, va='top', transform=axt.transAxes)
            y -= 0.19 + 0.15 * wrapped.count('\n')
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)
    out = os.path.join(FIG_DIR, 'qualitative.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out, ids, {i: {'gt': refs[i], 'ce': ce[i]['caption'], 'mixed': mixed[i]['caption'],
                          'pure': pure[i]['caption']} for i in ids}


@torch.no_grad()
def greedy_with_attention(head, x, word_map, max_len=20):
    """Greedy decoding that also returns the attention map of every generated word."""
    dec = head.decoder
    h, c = dec.init_hidden_state(x)
    prev = torch.full((1,), word_map['<start>'], dtype=torch.long, device=x.device)
    words, alphas = [], []
    for _ in range(max_len):
        context, alpha = dec.attention(x, h)
        context = dec.sigmoid(dec.f_beta(h)) * context
        h, c = dec.decode_step(torch.cat([dec.embedding(prev), context], dim=1), (h, c))
        nxt = dec.fc(h).argmax(dim=1)
        if nxt.item() == word_map['<end>']:
            break
        words.append(nxt.item())
        alphas.append(alpha[0].view(7, 7).cpu().numpy())
        prev = nxt
    return words, alphas


def attention_maps(n_images=2, n_words=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(WORD_MAP) as f:
        word_map = json.load(f)
    rev = {v: k for k, v in word_map.items()}
    run = os.path.join(BASE_DIR, 'runs', 'scst_mixed_se_s0')
    ck = torch.load(os.path.join(run, 'best.pt'), map_location=device, weights_only=False)
    head = CaptionHead(ck['feat_dim'], len(word_map)).to(device)
    head.load_state_dict(ck['model'])
    head.eval()
    feat_dir = os.path.join(BASE_DIR, ck['args']['feat_dir'])
    feats = np.load(os.path.join(feat_dir, 'test_feats.npy'), mmap_mode='r')
    with open(os.path.join(os.path.dirname(os.path.normpath(feat_dir)), 'test_meta.json')) as f:
        meta = json.load(f)
    idx = random.Random(SEED + 1).sample(range(len(meta)), n_images)

    fig, axes = plt.subplots(n_images, n_words + 1, figsize=(3.45, 1.05 * n_images + 0.3))
    captions = {}
    for r, i in enumerate(idx):
        x = head.proj(torch.from_numpy(np.asarray(feats[i], dtype=np.float32)).unsqueeze(0).to(device))
        words, alphas = greedy_with_attention(head, x, word_map)
        tokens = [rev[w] for w in words]
        captions[meta[i]['cocoid']] = ' '.join(tokens)
        img = np.asarray(load_image(meta[i]['filename'])).astype(np.float32) / 255.
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(textwrap.fill(' '.join(tokens), 27), fontsize=6, loc='left')
        # first content words, in caption order
        picks = [t for t, w in enumerate(tokens) if w not in DANGLING][:n_words]
        for k in range(n_words):
            ax = axes[r, k + 1]
            ax.axis('off')
            if k >= len(picks):
                continue
            t = picks[k]
            a = torch.from_numpy(alphas[t])[None, None]
            a = F.interpolate(a, size=(256, 256), mode='bilinear', align_corners=False)[0, 0].numpy()
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            # brightness follows attention: one-hue sequential encoding on top of the photo
            ax.imshow(img * (0.3 + 0.7 * a[..., None]))
            ax.set_title(f'"{tokens[t]}"', fontsize=7.5)
        axes[r, 0].axis('off')
    fig.tight_layout(pad=0.2, h_pad=0.9, w_pad=0.2)
    out = os.path.join(FIG_DIR, 'attention_maps.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out, captions


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    print(training_dynamics())
    out, ids, caps = qualitative()
    print(out)
    print(json.dumps(caps, indent=1))
    out, caps = attention_maps()
    print(out)
    print(json.dumps(caps, indent=1))
