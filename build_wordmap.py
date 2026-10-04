# build_wordmap.py
# Rebuilds WORDMAP_*.json from the Karpathy split JSON alone (no images needed).
# Mirrors the vocabulary logic in utils.create_input_files exactly.
import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
karpathy_json_path = os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'dataset_coco.json')
output_folder = os.path.join(BASE_DIR, 'data', 'coco', 'processed')
min_word_freq = 5

if __name__ == '__main__':
    with open(karpathy_json_path, 'r', encoding='utf-8') as j:
        data = json.load(j)

    word_freq = Counter()
    for img in data['images']:
        for c in img['sentences']:
            word_freq.update(c['tokens'])

    words = [w for w in word_freq if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    os.makedirs(output_folder, exist_ok=True)
    out = os.path.join(output_folder, f'WORDMAP_coco_5_cap_per_img_{min_word_freq}_min_word_freq.json')
    with open(out, 'w') as j:
        json.dump(word_map, j)
    print(f"Wrote {out} ({len(word_map)} entries)")
