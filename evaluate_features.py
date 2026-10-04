# evaluate_features.py
# Test-split evaluation for models trained by train_decoder.py on cached, frozen
# encoder features. Uses the same beam search, references (all raw captions,
# PTB tokenizer) and scoring as evaluate_coco.py, so numbers are comparable.
#
# Usage: python evaluate_features.py runs/scst_mixed_se_s0 [--beam 3 --alpha 0 --min_len 0 --block_ngram 0]
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from evaluate_coco import BASE_DIR, add_java_to_path, beam_search_from_features, caption_stats, score
from train_decoder import WORD_MAP, CaptionHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', help='run directory containing best.pt and config.json')
    p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--beam', type=int, default=7)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--min_len', type=int, default=5)
    p.add_argument('--block_ngram', type=int, default=4)
    p.add_argument('--max_steps', type=int, default=50)
    p.add_argument('--karpathy_json', default=os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json'))
    p.add_argument('--spice', action='store_true')
    args = p.parse_args()

    add_java_to_path(None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(WORD_MAP) as f:
        word_map = json.load(f)
    rev = {v: k for k, v in word_map.items()}

    ck = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location=device, weights_only=False)
    feat_dir = ck['args']['feat_dir']
    head = CaptionHead(ck['feat_dim'], len(word_map)).to(device)
    head.load_state_dict(ck['model'])
    head.eval()

    feats = np.load(os.path.join(BASE_DIR, feat_dir, f'{args.split}_feats.npy'), mmap_mode='r')
    meta_path = os.path.join(BASE_DIR, feat_dir, f'{args.split}_meta.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(os.path.normpath(os.path.join(BASE_DIR, feat_dir))), f'{args.split}_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    with open(args.karpathy_json, encoding='utf-8') as f:
        raw = {e['cocoid']: [s['raw'] for s in e['sentences']] for e in json.load(f)['images'] if e['split'] == args.split}

    gts, res, records = {}, {}, []
    t0 = time.time()
    with torch.no_grad():
        for i, m in enumerate(tqdm(meta, desc=f'{os.path.basename(args.run_dir)} beam={args.beam}')):
            x = head.proj(torch.from_numpy(np.asarray(feats[i], dtype=np.float32)).unsqueeze(0).to(device))
            words = beam_search_from_features(head.decoder, x, word_map, args.beam, args.alpha, args.min_len,
                                              args.block_ngram, args.max_steps, device)
            caption = ' '.join(rev[w] for w in words)
            gts[m['cocoid']] = raw[m['cocoid']]
            res[m['cocoid']] = caption
            records.append({'image_id': m['cocoid'], 'file': m['filename'], 'caption': caption})
    print(f"Decoded {len(records)} images in {time.time() - t0:.0f}s")

    metrics = score(gts, res, args.spice)
    stats = caption_stats([r['caption'] for r in records])
    tag = f"{os.path.basename(os.path.normpath(args.run_dir))}_{args.split}_b{args.beam}_a{args.alpha}_ml{args.min_len}_ng{args.block_ngram}"
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    out = os.path.join(BASE_DIR, 'results', f'eval_{tag}.json')
    with open(out, 'w') as f:
        json.dump({'config': vars(args), 'run_dir': args.run_dir, 'checkpoint_epoch': ck['epoch'],
                   'checkpoint_val': ck.get('val'), 'n_images': len(records), 'metrics': metrics,
                   'caption_stats': stats, 'captions': records}, f, indent=1)
    print(json.dumps({'metrics': metrics, 'caption_stats': stats}, indent=1))
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
