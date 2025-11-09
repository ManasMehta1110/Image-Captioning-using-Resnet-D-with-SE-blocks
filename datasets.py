import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Adapted for Transformer: captions are pre-padded, caplen used for masks/loss if needed.
    But since padded, padding mask can be derived from (caption == pad_idx).
    Fixed potential IndexError in all_captions slice for edge cases (e.g., incomplete last batch).
    Also, use == for string comparison to avoid SyntaxWarning in some linters/Pylance.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # Optional: Assert exact multiples for safety (comment out if your data has incomplete batches)
        # assert self.dataset_size % self.cpi == 0, f"Dataset size {self.dataset_size} not divisible by cpi {self.cpi}; check data preparation."

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image    
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':  # Changed from 'is' to '==' to avoid Pylance/SyntaxWarning
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            # Fixed slice to prevent IndexError if last image has fewer captions
            image_idx = i // self.cpi
            start = image_idx * self.cpi
            end = min(start + self.cpi, len(self.captions))
            all_captions = torch.LongTensor(self.captions[start:end])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size