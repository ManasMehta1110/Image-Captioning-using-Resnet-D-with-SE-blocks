import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import CaptionDataset  # Updated import to match file name
from utils import *  # Assuming this has any utils needed; adjust if not
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json  # Added for word_map loading
import torch.backends.cudnn as cudnn

# Parameters
data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = '../BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = '/media/ssd/caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model (adapted for CaptionModel or separate encoder/decoder)
checkpoint = torch.load(checkpoint, map_location=str(device))
if 'model' in checkpoint:
    model = checkpoint['model']
    decoder = model.decoder
    encoder = model.encoder
else:
    # Fallback to original style
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
decoder = decoder.to(device)
decoder.eval()
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform (input size 256 ok for ResNet50; for 224, add Resize(224) below if regenerating data)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def generate_square_subsequent_mask(sz):
    """Generates a square subsequent mask for the Transformer decoder."""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)
    return mask


def beam_search_transformer(encoder_out, decoder, k, max_len=50, sos_idx=None, eos_idx=None):
    """
    Beam search for Transformer decoder.
    
    :param encoder_out: (1, num_patches, d_model) encoded features
    :param decoder: TransformerDecoder
    :param k: beam size
    :param max_len: max sequence length
    :param sos_idx: start token index
    :param eos_idx: end token index
    :return: best sequence (list of ints)
    """
    # Expand encoder_out for beams (though shared)
    encoder_out = encoder_out.expand(k, -1, -1)  # (k, num_patches, d_model)

    # Init with <start>
    seqs = torch.full((k, 1), sos_idx, dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, 1, device=device)

    complete_seqs = []
    complete_seqs_scores = []

    for step in range(1, max_len + 1):
        current_len = seqs.size(1)
        tgt_input = seqs  # (k, current_len)

        # Causal mask
        tgt_mask = generate_square_subsequent_mask(current_len).unsqueeze(0).expand(k, -1, -1)

        # Forward pass
        outputs = decoder(tgt_input, encoder_out, tgt_mask=tgt_mask)  # (k, current_len, vocab_size)
        scores = F.log_softmax(outputs[:, -1, :], dim=1)  # (k, vocab_size)
        scores = top_k_scores + scores  # Cumulative

        # Top k across all
        flattened_scores, flattened_indices = scores.view(-1).topk(k, 0, True, True)
        beam_indices = flattened_indices // vocab_size
        next_word_indices = flattened_indices % vocab_size

        # Update seqs
        new_seqs = torch.cat([seqs[beam_indices], next_word_indices.unsqueeze(1)], dim=1)
        new_scores = flattened_scores.unsqueeze(1)

        # Check EOS
        eos_mask = next_word_indices == eos_idx
        if eos_mask.any():
            # Collect completed
            completed_beams = beam_indices[eos_mask]
            for idx in range(eos_mask.sum().item()):
                comp_beam = completed_beams[idx].item()
                complete_seqs.append(new_seqs[idx].tolist())
                complete_seqs_scores.append(new_scores[idx].item())
            # Keep only non-EOS
            keep_mask = ~eos_mask
            if keep_mask.sum() == 0:
                break
            beam_indices = beam_indices[keep_mask]
            next_word_indices = next_word_indices[keep_mask]
            new_seqs = new_seqs[keep_mask]
            new_scores = new_scores[keep_mask]
            k_active = keep_mask.sum().item()
        else:
            k_active = k

        # Update
        seqs = new_seqs
        top_k_scores = new_scores

    # Best sequence
    if complete_seqs:
        best_idx = np.argmax(complete_seqs_scores)
        return complete_seqs[best_idx]
    else:
        # Fallback to best unfinished
        best_beam = top_k_scores.argmax().item()
        return seqs[best_beam].tolist()


def evaluate(beam_size):
    """
    Evaluation for Transformer model.

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader (batch_size=1 for simplicity; transform includes normalize)
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        # Move to device
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        with torch.no_grad():
            encoder_out = encoder(image)  # (1, 49, d_model)

        # Generate with beam search
        seq = beam_search_transformer(encoder_out, decoder, beam_size, max_len=50,
                                      sos_idx=word_map['<start>'], eos_idx=word_map['<end>'])

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start>, <end>, <pad>
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    import numpy as np  # Added for argmax
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))