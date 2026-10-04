# make_diagrams.py
# Vector diagrams for the method section:
#   paper/figs/architecture.pdf - frozen encoder -> projection -> attention LSTM decoder
#   paper/figs/stem.pdf         - ResNet vs ResNet-D stem and downsampling shortcut
#   paper/figs/se_block.pdf     - Squeeze-and-Excitation block
# Frozen (pretrained) parts are gray and labeled "frozen"; trained parts are blue and
# labeled "trained", so the distinction never rests on color alone.
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'paper', 'figs')

INK, MUTED = '#1f1f1e', '#6b6a63'
FROZEN_FILL, FROZEN_EDGE = '#eeede8', '#8a897f'
TRAIN_FILL, TRAIN_EDGE = '#dceafa', '#2a78d6'
DATA_FILL, DATA_EDGE = '#ffffff', '#8a897f'

plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'text.color': INK,
                     'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})


def box(ax, x, y, w, h, text, kind='data', fs=6.5, bold=False):
    fill, edge = {'frozen': (FROZEN_FILL, FROZEN_EDGE), 'train': (TRAIN_FILL, TRAIN_EDGE),
                  'data': (DATA_FILL, DATA_EDGE)}[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                facecolor=fill, edgecolor=edge, linewidth=0.8))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal',
            linespacing=1.15)


def arrow(ax, p, q, rad=0.0, style='-|>', color=MUTED, lw=0.8, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, linewidth=lw,
                                 linestyle=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def canvas(w_in, h_in, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis('off')
    return fig, ax


def save(fig, name):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.pdf', '.png'), dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return out


def architecture():
    fig, ax = canvas(6.9, 2.45, 215, 74)
    # input image: first Karpathy test image, resized as in preprocessing
    with open(os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json'), encoding='utf-8') as f:
        e = next(x for x in json.load(f)['images'] if x['split'] == 'test')
    img = Image.open(os.path.join(BASE_DIR, 'data', 'coco', 'images', e['filepath'], e['filename'])).convert('RGB')
    ax.imshow(img.resize((256, 256)), extent=(1, 23, 33, 55), zorder=2)
    ax.text(12, 30, 'image\n$256\\times256$', ha='center', va='top', fontsize=6)

    # encoder side
    box(ax, 44, 44, 34, 26, 'CNN encoder\nResNet-50 /\nSE-ResNet-50 /\nResNet-50-D\n(ImageNet, frozen)', 'frozen', fs=6)
    arrow(ax, (24, 44), (27, 44))
    box(ax, 79, 44, 28, 18, '$8\\times8\\times2048$\navg-pool to\n$7\\times7\\times2048$', 'data', fs=6)
    arrow(ax, (61, 44), (65, 44))
    box(ax, 109, 44, 24, 18, 'linear\n$2048\\to512$\n(trained)', 'train', fs=6)
    arrow(ax, (93, 44), (97, 44))
    box(ax, 109, 15, 32, 14, '$\\mathbf{a}_1,\\dots,\\mathbf{a}_{49}$\n49 vectors in $\\mathbb{R}^{512}$', 'data', fs=6)
    arrow(ax, (109, 35), (109, 22))

    # decoder side, one time step
    ax.add_patch(FancyBboxPatch((130, 3), 83, 69, boxstyle='round,pad=0,rounding_size=2',
                                facecolor='none', edgecolor=TRAIN_EDGE, linewidth=0.8, linestyle=(0, (3, 2))))
    ax.text(133, 69.5, 'attention LSTM decoder, step $t$ (trained)', ha='left', va='top', fontsize=6,
            color=TRAIN_EDGE)
    box(ax, 150, 15, 26, 13, 'attention\n$\\alpha_{t,k}$, $\\hat{\\mathbf{z}}_t$', 'train', fs=6)
    arrow(ax, (125, 15), (137, 15))
    box(ax, 150, 37, 26, 13, 'gate\n$\\boldsymbol{\\beta}_t\\odot\\hat{\\mathbf{z}}_t$', 'train', fs=6)
    arrow(ax, (150, 21.5), (150, 30.5))
    box(ax, 150, 57, 26, 11, 'embedding\n$\\mathbf{E}y_{t-1}$', 'train', fs=6)
    box(ax, 186, 44, 22, 16, 'LSTM\n$\\mathbf{h}_t,\\mathbf{c}_t$', 'train', fs=6.5)
    arrow(ax, (163, 38), (175, 42))
    arrow(ax, (163, 56), (175, 47))
    box(ax, 186, 15, 26, 13, 'linear +\nsoftmax\n$p(y_t)$', 'train', fs=6)
    arrow(ax, (186, 36), (186, 21.5))
    arrow(ax, (199, 15), (205, 15))
    ax.text(209, 15, '$y_t$', ha='center', va='center', fontsize=8)
    # recurrent connection: h_t is used by attention and gate at the next step
    arrow(ax, (197, 44), (203, 44), style='-')
    arrow(ax, (203, 44), (203, 28), style='-')
    arrow(ax, (203, 28), (163, 28), style='-|>', ls=(0, (2, 1.5)))
    ax.text(183, 26.5, '$\\mathbf{h}_{t}$ used at step $t{+}1$', ha='right', va='top', fontsize=5.5, color=MUTED)

    # legend
    # legend, stacked in the empty area under the encoder
    box(ax, 32, 15, 6, 4, '', 'frozen')
    ax.text(37, 15, 'frozen (pretrained)', va='center', fontsize=6)
    box(ax, 32, 7, 6, 4, '', 'train')
    ax.text(37, 7, 'trained', va='center', fontsize=6)
    return save(fig, 'architecture.pdf')


def stem():
    fig, ax = canvas(3.45, 3.0, 100, 100)
    ax.text(25, 98, '(a) stem', ha='center', va='top', fontsize=7, fontweight='bold')
    ax.text(25, 92, 'ResNet', ha='center', va='top', fontsize=6.5, color=MUTED)
    ax.text(75, 92, 'ResNet-D', ha='center', va='top', fontsize=6.5, color=MUTED)
    w, h = 46, 7.5
    left = [(80, '$7\\times7$ conv, 64, stride 2'), (61, '$3\\times3$ max pool, stride 2')]
    right = [(84, '$3\\times3$ conv, 32, stride 2'), (75, '$3\\times3$ conv, 32'), (66, '$3\\times3$ conv, 64'),
             (57, '$3\\times3$ max pool, stride 2')]
    for col, steps in ((25, left), (75, right)):
        for i, (y, t) in enumerate(steps):
            box(ax, col, y, w, h, t, 'data', fs=6)
            if i:
                arrow(ax, (col, steps[i - 1][0] - h / 2), (col, y + h / 2))
    ax.text(50, 47, 'each convolution is followed by batch normalization and ReLU', ha='center', fontsize=5.8,
            color=MUTED)

    ax.text(50, 43.5, '(b) downsampling shortcut', ha='center', va='top', fontsize=7, fontweight='bold')
    box(ax, 25, 26, w, h, '$1\\times1$ conv, stride 2', 'data', fs=6)
    box(ax, 75, 31, w, h, '$2\\times2$ avg pool, stride 2', 'data', fs=6)
    box(ax, 75, 18, w, h, '$1\\times1$ conv, stride 1', 'data', fs=6)
    arrow(ax, (75, 31 - h / 2), (75, 18 + h / 2))
    for col, top, bottom in ((25, 26 + h / 2, 26 - h / 2), (75, 31 + h / 2, 18 - h / 2)):
        arrow(ax, (col, 38.5), (col, top))
        arrow(ax, (col, bottom), (col, 8))
    ax.text(25, 5, 'skips 3 of every 4 inputs', ha='center', fontsize=5.8, color=MUTED)
    ax.text(75, 5, 'averages all inputs', ha='center', fontsize=5.8, color=MUTED)
    return save(fig, 'stem.pdf')


def se_block():
    fig, ax = canvas(3.45, 1.55, 100, 44)
    steps = [(9, 'input $\\mathbf{U}$\n$C\\times H\\times W$', 'data', 17),
             (28, 'global\navg pool', 'data', 15),
             (47, 'FC\n$C\\to C/r$\nReLU', 'data', 17),
             (66, 'FC\n$C/r\\to C$\nsigmoid', 'data', 17),
             (84, 'scale\n$s_c\\,\\mathbf{u}_c$', 'data', 13)]
    y = 26
    for i, (x, t, kind, w) in enumerate(steps):
        box(ax, x, y, w, 17, t, kind, fs=5.8)
        if i:
            px, _, _, pw = steps[i - 1]
            arrow(ax, (px + pw / 2, y), (x - w / 2, y))
    # identity path: U is rescaled channel-wise
    arrow(ax, (9, y - 8.5), (9, 6), style='-')
    arrow(ax, (9, 6), (84, 6), style='-')
    arrow(ax, (84, 6), (84, y - 8.5))
    ax.text(47, 7.2, '$\\mathbf{U}$', ha='center', va='bottom', fontsize=6.5, color=MUTED)
    arrow(ax, (90.5, y), (94.5, y))
    ax.text(97.5, y, '$\\tilde{\\mathbf{U}}$', ha='center', va='center', fontsize=8)
    ax.text(28, 41, 'squeeze', ha='center', fontsize=6.5, color=MUTED)
    ax.text(56.5, 41, 'excitation', ha='center', fontsize=6.5, color=MUTED)
    ax.text(84, 41, 'scale', ha='center', fontsize=6.5, color=MUTED)
    return save(fig, 'se_block.pdf')


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    for f in (architecture, stem):
        print(f())
