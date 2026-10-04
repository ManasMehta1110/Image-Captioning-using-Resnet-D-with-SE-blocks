# cache_features.py
# Runs the (frozen) encoder of a checkpoint once over Karpathy splits and caches
# its 49x512 outputs, so decoder-only fine-tuning runs fast on a small GPU.
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_coco import BASE_DIR, TestImages


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=os.path.join(
        BASE_DIR, 'checkpoint_coco_5_cap_per_img_5_min_word_freq_epoch_87.pth.tar'))
    p.add_argument('--karpathy_json', default=os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json'))
    p.add_argument('--image_root', default=os.path.join(BASE_DIR, 'data', 'coco', 'images'))
    p.add_argument('--out_dir', default=os.path.join(BASE_DIR, 'data', 'coco', 'features'))
    p.add_argument('--splits', nargs='+', default=['restval', 'val', 'test'])
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = ckpt['encoder'].to(device).eval()

    with open(args.karpathy_json, encoding='utf-8') as f:
        images = json.load(f)['images']
    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        entries = [e for e in images if e['split'] == split]
        # Only images present on disk (restval/val/test all come from val2014)
        entries = [e for e in entries if os.path.exists(os.path.join(args.image_root, e['filepath'], e['filename']))]
        feats = np.lib.format.open_memmap(os.path.join(args.out_dir, f'{split}_feats.npy'), mode='w+',
                                          dtype=np.float16, shape=(len(entries), 49, 512))
        loader = DataLoader(TestImages(entries, args.image_root), batch_size=args.batch_size,
                            shuffle=False, num_workers=6, pin_memory=True)
        with torch.no_grad():
            for imgs, idx in tqdm(loader, desc=split):
                feats[idx.numpy()] = encoder(imgs.to(device)).cpu().numpy().astype(np.float16)
        feats.flush()

        meta = [{'cocoid': e['cocoid'], 'filename': e['filename'],
                 'tokens': [s['tokens'] for s in e['sentences']]} for e in entries]
        with open(os.path.join(args.out_dir, f'{split}_meta.json'), 'w') as f:
            json.dump(meta, f)
        print(f"{split}: {len(entries)} images cached")


if __name__ == '__main__':
    main()
