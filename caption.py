import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import resize  # Updated from deprecated scipy.misc
import argparse
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_square_subsequent_mask(sz):
    """Generates a square subsequent mask for the decoder."""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)
    return mask


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3, max_len=50):
    """
    Reads an image and captions it with beam search for Transformer decoder.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :param max_len: maximum caption length
    :return: caption, alphas for visualization (averaged cross-attention to image regions)
    """

    k = beam_size
    vocab_size = len(word_map)
    sos_idx = word_map['<start>']
    eos_idx = word_map['<end>']
    pad_idx = word_map['<pad>']

    # Read image and process (updated to 224x224 for ResNet-50)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Standard for ResNet-50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Encode
    encoder.eval()
    with torch.no_grad():
        encoder_out = encoder(image)  # (1, 49, d_model)  # 7x7=49 patches
    num_pixels = encoder_out.size(1)  # 49
    d_model = encoder_out.size(2)

    # Expand encoder_out for k beams
    encoder_out = encoder_out.expand(k, num_pixels, d_model)  # (k, 49, d_model)

    # Tensor to store top k previous sequences; init with <start>
    seqs = torch.full((k, 1), sos_idx, dtype=torch.long, device=device)  # (k, 1)

    # Tensor to store top k scores; init to 0
    top_k_scores = torch.zeros(k, 1, device=device)  # (k, 1)

    # Lists for completed sequences, scores, and alphas
    complete_seqs = []
    complete_seqs_scores = []
    complete_seqs_alphas = []  # Will store list of (step, alphas)

    # Start decoding
    for step in range(1, max_len + 1):
        # For each beam, prepare input sequence up to current length
        # But since beams have different lengths? No, in standard beam, all active beams have same length; completed are set aside
        # Here, we process all k, then filter

        # Current input for decoder: seqs (k, step) but actually, for efficiency, we can use a single forward with padding, but for simplicity, process per beam? Wait, better: since all active have same len, but to simplify, use loop over beams? No, batch them.

        # Prepare tgt: seqs[:, :step] but since step increases, and seqs grows
        # Actually, standard way: at each step, input is previous seqs (k, current_len), predict next

        current_len = seqs.size(1)
        tgt_input = seqs  # (k, current_len)

        # Generate mask for causal attention
        tgt_mask = generate_square_subsequent_mask(current_len).unsqueeze(0).expand(k, -1, -1)  # (k, len, len)

        # No padding mask since no pads during generation

        decoder.eval()
        with torch.no_grad():
            outputs = decoder(tgt_input, encoder_out, tgt_mask=tgt_mask)  # (k, current_len, vocab_size)

        # Scores for next token: log softmax on last position
        scores = F.log_softmax(outputs[:, -1, :], dim=1)  # (k, vocab_size)

        # Add to previous scores
        scores = top_k_scores + scores  # (k, vocab_size)  # Broadcasting

        # Find top k candidates across all beams (flatten)
        flattened_scores, flattened_indices = scores.view(-1).topk(k, 0, True, True)  # (k,)

        # Get beam indices and word indices
        beam_indices = flattened_indices / vocab_size  # (k,)
        next_word_indices = flattened_indices % vocab_size  # (k,)

        # Update sequences and scores
        new_seqs = torch.cat([seqs[beam_indices], next_word_indices.unsqueeze(1)], dim=1)  # (k, current_len+1)
        new_scores = flattened_scores.unsqueeze(1)  # (k, 1)

        # Check for completions (eos)
        eos_mask = next_word_indices == eos_idx
        if eos_mask.any():
            # Add completed to lists
            completed_beams = beam_indices[eos_mask]
            for i in range(eos_mask.sum().item()):
                comp_idx = completed_beams[i].item()
                complete_seqs.append(new_seqs[i].tolist())
                complete_seqs_scores.append(new_scores[i].item())
                # For alphas: need to extract attention; but for now, we'll compute separately later for best seq
                # Placeholder: append None
                complete_seqs_alphas.append(None)

            # Remove completed from active
            keep_mask = ~eos_mask
            if keep_mask.sum() == 0:
                break
            beam_indices = beam_indices[keep_mask]
            next_word_indices = next_word_indices[keep_mask]
            new_seqs = new_seqs[keep_mask]
            new_scores = new_scores[keep_mask]
            k = keep_mask.sum().item()

        # Update for next iteration
        seqs = new_seqs
        top_k_scores = new_scores
        encoder_out = encoder_out[beam_indices]  # Update encoder_out? Wait, no: encoder_out is shared, but if we want per-beam, but actually it's the same for all beams

        # Actually, since memory is same for all, no need to index encoder_out

    # If no complete, take best unfinished
    if not complete_seqs:
        i = top_k_scores.argmax()
        seq = seqs[i].tolist()
        score = top_k_scores[i].item()
    else:
        i = np.argmax(complete_seqs_scores)
        seq = complete_seqs[i]
        score = complete_seqs_scores[i]

    # Now, for alphas: re-run generation for best seq to extract cross-attention
    # To get alphas, we'll do greedy re-generation for the best seq and extract attentions
    alphas = extract_attention_alphas(encoder, decoder, image_path, seq[:-1], word_map, encoder_out[0:i])  # Use first encoder_out as approx

    return seq, alphas


def extract_attention_alphas(encoder, decoder, image_path, seq, word_map, encoder_out, grid_size=7):
    """
    Extracts averaged cross-attention weights to image regions for visualization.
    Runs greedy forward on the sequence to get attentions from Transformer decoder.
    """
    # Note: This is approximate; for exact beam path, would need to track during beam, but complex.
    # Averages over heads and layers for simplicity.

    vocab_size = len(word_map)
    sos_idx = word_map['<start>']
    seq = [sos_idx] + seq  # Full input seq

    # Prepare image (but we have encoder_out, reuse)
    # Assume encoder_out is (1, 49, d_model); we'll use it

    encoder_out = encoder_out.unsqueeze(0)  # Ensure (1, 49, d_model)

    # To extract attention, we need to access decoder's transformer_decoder layers
    # Monkey patch or modify decoder to return attentions; for this, we'll add a flag or hook
    # Simple way: temporarily modify decoder to return attentions

    # For demo, we'll implement a forward that returns attentions
    # Assuming decoder.transformer_decoder.layers are accessible

    attentions = []
    decoder.eval()

    # Build input step by step (greedy, but since seq known, can do full forward)
    tgt_input = torch.tensor([seq], dtype=torch.long, device=device)  # (1, len(seq))
    tgt_len = tgt_input.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_len).unsqueeze(0)

    # Hook to capture cross-attention (memory -> query)
    # We'll use a simple forward and assume we can access layer.self_attn or wait, for decoder, it's multihead_attention in layer

    # To make it work, let's define a wrapper
    def forward_with_attention(tgt_emb, memory):
        # tgt_emb: (tgt_len, 1, d_model)
        # This is manual unroll of transformer_decoder to capture attn_output_weights from multiheadattention

        # Assume decoder has num_layers=3, etc.; but to general, we'll recurse or loop over layers

        # For simplicity, we'll only capture from the last layer's cross-attention
        # First, run full decoder to get output (but we don't need)
        # Use torch hooks

        cross_attns = [None] * decoder.transformer_decoder.num_layers

        def hook_fn(module, input, output):
            # For TransformerDecoderLayer, output[1] is attn_output_weights for self, output[2] for cross
            if hasattr(module, 'multihead_attn'):  # cross
                cross_attns[decoder.transformer_decoder.layers.index(module)] = output[2]  # (batch, tgt_len, src_len)

        # Register hooks on cross-attention modules
        handles = []
        for layer in decoder.transformer_decoder.layers:
            handle = layer.multihead_attn.register_forward_hook(hook_fn)
            handles.append(handle)

        # Run forward
        decoder(tgt_input, memory, tgt_mask=tgt_mask)

        # Remove hooks
        for h in handles:
            h.remove()

        # Average over layers and heads: cross_attns[l] is (1, tgt_len, 49, num_heads)? Wait, actually for nn.MultiheadAttention, attn_output_weights is (batch*num_heads, tgt_len, src_len)
        # Standard: (N*num_heads, L, S)

        # For each layer, average over heads: mean over dim=0 after reshape
        layer_alphas = []
        for attn in cross_attns:
            if attn is not None:
                # attn: (1*heads, tgt_len, 49)
                attn = attn.mean(dim=0)  # Avg over heads: (tgt_len, 49)
                layer_alphas.append(attn)

        # Avg over layers
        if layer_alphas:
            avg_alpha = torch.stack(layer_alphas).mean(0)  # (tgt_len, 49)
            alphas = avg_alpha[1:].cpu().numpy()  # Skip <start>, (len-1, 49)
        else:
            alphas = np.zeros((len(seq)-1, 49))  # Fallback

        return alphas

    # Get tgt_emb? Wait, no: decoder forward takes token ids, embeds inside
    # So call decoder(tgt_input, encoder_out, tgt_mask) but since it's modified, it runs

    # Actually, in the hook setup above, we call decoder(...) inside the def, but to extract:
    # Move the call outside

    # Corrected:
    # Setup hooks first
    cross_attns = [None] * decoder.transformer_decoder.num_layers

    def hook_fn(module, input, output):
        layer_idx = decoder.transformer_decoder.layers.index(module)
        cross_attns[layer_idx] = output[2]  # Cross-attn weights

    handles = []
    for layer in decoder.transformer_decoder.layers:
        handle = layer.multihead_attn.register_forward_hook(hook_fn)
        handles.append(handle)

    # Run
    _ = decoder(tgt_input, encoder_out, tgt_mask=tgt_mask)  # Dummy output

    # Cleanup
    for h in handles:
        h.remove()

    # Process as above
    layer_alphas = []
    for attn in cross_attns:
        if attn is not None:
            # attn shape: (batch * heads, tgt_len, memory_len) = (1*8, tgt_len, 49)
            attn = attn.view(1, decoder.transformer_decoder.layers[0].multihead_attn.num_heads, tgt_len, 49)
            attn = attn.mean(dim=1)  # Avg heads: (1, tgt_len, 49)
            attn = attn.squeeze(0)  # (tgt_len, 49)
            layer_alphas.append(attn)

    if layer_alphas:
        avg_alpha = torch.stack(layer_alphas).mean(0)  # (tgt_len, 49)
        alphas = avg_alpha[1:].cpu().numpy()  # Skip first (<start>), shape (seq_len-1, 49)
    else:
        alphas = np.zeros((len(seq)-1, 49))

    return alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True, grid_size=7):
    """
    Visualizes caption with weights at every word for 7x7 grid.

    Adapted from original.

    :param image_path: path to image
    :param seq: caption (list of indices, including <start>)
    :param alphas: weights (num_words, num_pixels=49)
    :param rev_word_map: reverse word mapping
    :param smooth: smooth weights?
    :param grid_size: spatial grid size (7 for ResNet50)
    """
    image = Image.open(image_path)
    image = image.resize([grid_size * 24, grid_size * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq if ind in rev_word_map]  # Skip special tokens if needed
    words = words[1:]  # Skip <start>

    for t in range(len(words)):
        if t >= len(alphas) or t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :].reshape(grid_size, grid_size)  # (7,7)
        if smooth:
            alpha = resize(current_alpha, (grid_size * 24, grid_size * 24), mode='reflect')  # Updated resize
        else:
            alpha = resize(current_alpha, (grid_size * 24, grid_size * 24))
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Captioning - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image', required=True)
    parser.add_argument('--model', '-m', help='path to model', required=True)
    parser.add_argument('--word_map', '-wm', help='path to word map JSON', required=True)
    parser.add_argument('--beam_size', '-b', default=3, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--max_len', default=50, type=int, help='max caption length')

    args = parser.parse_args()

    # Load model (assumes checkpoint has 'model' as CaptionModel; adjust if saved as encoder/decoder separately)
    checkpoint = torch.load(args.model, map_location=str(device))
    if 'model' in checkpoint:
        model = checkpoint['model']
        encoder = model.encoder
        decoder = model.decoder
    else:
        # Fallback to original style
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    decoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Encode, decode with beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size, args.max_len)

    # Create caption string (exclude <start>, stop at <end>)
    caption = [rev_word_map[idx] for idx in seq if idx != word_map['<start>'] and idx != word_map['<end>']]
    print('Generated caption:', ' '.join(caption))

    # Visualize if alphas available
    if alphas is not None and len(alphas) > 0:
        alphas_tensor = torch.FloatTensor(alphas)
        visualize_att(args.img, seq, alphas_tensor, rev_word_map, args.smooth, grid_size=7)
    else:
        print('Attention visualization skipped (no alphas extracted).')