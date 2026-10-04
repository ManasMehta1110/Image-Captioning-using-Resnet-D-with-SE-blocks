# cache_timm_features.py
# Encoder ablation: caches 7x7x2048 features from ImageNet-pretrained, frozen
# timm encoders trained with the SAME recipe (a1_in1k), so the only difference
# between them is the architectural modification:
#   resnet50   - baseline
#   seresnet50 - + Squeeze-and-Excitation blocks
#   resnet50d  - + ResNet-D (deep 3x3 stem + avg-pool downsampling shortcut)
# Each image is decoded once and passed through all encoders.
import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_coco import BASE_DIR, TestImages

ENCODERS = ['resnet50.a1_in1k', 'seresnet50.a1_in1k', 'resnet50d.a1_in1k']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--karpathy_json', default=os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json'))
    p.add_argument('--image_root', default=os.path.join(BASE_DIR, 'data', 'coco', 'images'))
    p.add_argument('--out_dir', default=os.path.join(BASE_DIR, 'data', 'coco', 'timm_features'))
    p.add_argument('--splits', nargs='+', default=['train', 'restval', 'val', 'test'])
    p.add_argument('--encoders', nargs='+', default=ENCODERS)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    for name in args.encoders:
        m = timm.create_model(name, pretrained=True).to(device).eval()
        cfg = m.pretrained_cfg
        # Same ImageNet normalization as TestImages uses; fail loudly if a model expects otherwise
        assert np.allclose(cfg['mean'], (0.485, 0.456, 0.406)) and np.allclose(cfg['std'], (0.229, 0.224, 0.225)), name
        models[name] = m

    with open(args.karpathy_json, encoding='utf-8') as f:
        images = json.load(f)['images']

    for split in args.splits:
        entries = [e for e in images if e['split'] == split]
        missing = [e for e in entries if not os.path.exists(os.path.join(args.image_root, e['filepath'], e['filename']))]
        assert not missing, f"{split}: {len(missing)} images missing on disk (e.g. {missing[0]['filename']})"

        feats = {}
        for name in args.encoders:
            d = os.path.join(args.out_dir, name.split('.')[0])
            os.makedirs(d, exist_ok=True)
            feats[name] = np.lib.format.open_memmap(os.path.join(d, f'{split}_feats.npy'), mode='w+',
                                                    dtype=np.float16, shape=(len(entries), 49, 2048))

        loader = DataLoader(TestImages(entries, args.image_root), batch_size=args.batch_size,
                            shuffle=False, num_workers=6, pin_memory=True)
        with torch.no_grad():
            for imgs, idx in tqdm(loader, desc=split):
                imgs = imgs.to(device)
                for name, m in models.items():
                    x = F.adaptive_avg_pool2d(m.forward_features(imgs), (7, 7))   # (B, 2048, 7, 7)
                    x = x.flatten(2).transpose(1, 2)                              # (B, 49, 2048)
                    feats[name][idx.numpy()] = x.cpu().numpy().astype(np.float16)
        for f in feats.values():
            f.flush()

        meta = [{'cocoid': e['cocoid'], 'filename': e['filename'],
                 'tokens': [s['tokens'] for s in e['sentences']]} for e in entries]
        with open(os.path.join(args.out_dir, f'{split}_meta.json'), 'w') as f:
            json.dump(meta, f)
        print(f"{split}: {len(entries)} images cached for {len(models)} encoders")


if __name__ == '__main__':
    main()
