import os
import numpy as np
import h5py
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder,
                       captions_per_image, min_word_freq,
                       output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # =====================================================
    # ✅ FIX: ensure output directory exists
    # =====================================================
    os.makedirs(output_folder, exist_ok=True)

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r', encoding='utf-8') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = (
            os.path.join(image_folder, img['filepath'], img['filename'])
            if dataset == 'coco'
            else os.path.join(image_folder, img['filename'])
        )

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] == 'val':
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] == 'test':
            test_image_paths.append(path)
            test_image_captions.append(captions)

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    words = [w for w in word_freq if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    base_filename = (
        f"{dataset}_{captions_per_image}_cap_per_img_"
        f"{min_word_freq}_min_word_freq"
    )

    with open(os.path.join(output_folder,
              f'WORDMAP_{base_filename}.json'), 'w') as j:
        json.dump(word_map, j)

    seed(123)

    for impaths, imcaps, split in [
        (train_image_paths, train_image_captions, 'TRAIN'),
        (val_image_paths, val_image_captions, 'VAL'),
        (test_image_paths, test_image_captions, 'TEST')
    ]:

        with h5py.File(
            os.path.join(output_folder,
                         f'{split}_IMAGES_{base_filename}.hdf5'), 'a'
        ) as h:

            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset(
                'images', (len(impaths), 3, 256, 256), dtype='uint8'
            )

            print(f"\nReading {split} images and captions...\n")

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [
                        choice(imcaps[i])
                        for _ in range(captions_per_image - len(imcaps[i]))
                    ]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                img = Image.open(path).convert('RGB')
                img = img.resize((256, 256), Image.LANCZOS)
                img = np.array(img).transpose(2, 0, 1)
                images[i] = img

                for c in captions:
                    enc_c = (
                        [word_map['<start>']] +
                        [word_map.get(w, word_map['<unk>']) for w in c] +
                        [word_map['<end>']] +
                        [word_map['<pad>']] * (max_len - len(c))
                    )
                    enc_captions.append(enc_c)
                    caplens.append(len(c) + 2)

            with open(os.path.join(
                output_folder, f'{split}_CAPTIONS_{base_filename}.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(
                output_folder, f'{split}_CAPLENS_{base_filename}.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')
        word = line[0]
        embedding = list(map(float, line[1:]))

        if word in vocab:
            embeddings[word_map[word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement,
                    encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):

    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer
    }

    filename = f'checkpoint_{data_name}.pth.tar'
    torch.save(state, filename)

    if is_best:
        torch.save(state, f'BEST_{filename}')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= shrink_factor
    print(f"New LR: {optimizer.param_groups[0]['lr']}\n")


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    return correct.view(-1).float().sum().item() * (100.0 / batch_size)
