import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper functions for ResNet (from torchvision style)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# SE-ResNet-D Bottleneck (integrates SE + ResNet-D downsampling refinements)
class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, se_ratio=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # conv1 is 1x1 (stride=1 always)
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # conv2 is 3x3 with stride (ResNet-B/V1.5 style)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # conv3 is 1x1 to expand
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, se_ratio)  # SE after conv3/bn3
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)  # Apply SE recalibration

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Custom ResNet-50-D + SE (refined stem + D downsampling)
class ResNetDSE(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, se_ratio=16):
        super(ResNetDSE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        # ResNet-D refined stem (from ResNet-C: three 3x3 convs instead of 7x7 + maxpool)
        self.conv1 = conv3x3(3, 32, stride=2)  # First 3x3 stride 2
        self.bn1 = norm_layer(32)
        self.conv2 = conv3x3(32, 32, stride=1)  # Second 3x3 stride 1
        self.bn2 = norm_layer(32)
        self.conv3 = conv3x3(32, 64, stride=1)  # Third 3x3 stride 1
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Retain maxpool

        self.layer1 = self._make_layer(block, 64, layers[0])  # No downsample (stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, se_ratio=se_ratio)  # Downsample
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, se_ratio=se_ratio)  # Downsample
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, se_ratio=se_ratio)  # Downsample

        # For captioning: No avgpool + FC; output spatial features from layer4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init the last BN in each residual branch (standard)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, se_ratio=16, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = 1
        if dilate:
            raise NotImplementedError("Dilation not supported in this implementation.")
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Use conv1x1 with stride to match residual branch exactly (avoid rounding mismatches)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # instantiate block with explicit keywords to avoid argument mismatch
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, se_ratio=se_ratio))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, se_ratio=se_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Refined stem forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)  # Now at 56x56 for 224 input

        x = self.layer1(x)  # 56x56 -> 256 channels
        x = self.layer2(x)  # 28x28 -> 512
        x = self.layer3(x)  # 14x14 -> 1024
        x = self.layer4(x)  # 7x7 -> 2048

        return x  # Shape: (batch, 2048, 7, 7)


class Encoder(nn.Module):
    """
    Encoder with ResNet50-D + SE.
    Outputs flattened spatial features projected to d_model.
    """

    def __init__(self, d_model=512, encoded_image_size=7):  # 7x7 for ResNet50
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Build ResNet-50-D + SE (layers=[3,4,6,3] for ResNet50)
        resnet = ResNetDSE(SEBottleneck, [3, 4, 6, 3], se_ratio=16)
        self.resnet = resnet

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Project to d_model (after flatten 7x7=49)
        self.img_weight = nn.Linear(2048, d_model)  # (batch, 49, 2048) -> (batch, 49, d_model)

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images (batch_size, num_pixels=49, d_model)
        """
        out = self.resnet(images)  # (batch_size, 2048, 7, 7)
        out = self.adaptive_pool(out)  # Ensure (batch_size, 2048, 7, 7)
        out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 2048)
        out = out.view(out.size(0), -1, out.size(3))  # (batch_size, 49, 2048)
        out = self.img_weight(out)  # (batch_size, 49, d_model)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4 (layer2,3,4)
        for c in [self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for p in c.parameters():
                p.requires_grad = fine_tune


# Positional Encoding for Transformer (standard sinusoidal)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # store as (max_len, 1, d_model) so it can broadcast over batch dim
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        Return: x + pos_encoding (seq_len, batch, d_model)
        If seq_len > stored max_len, compute positional encodings on the fly.
        """
        seq_len = x.size(0)
        d_model = x.size(2)
        if seq_len <= self.pe.size(0):
            return x + self.pe[:seq_len, :]  # broadcasts over batch dim
        else:
            # compute needed positional encodings on the fly (device-aware)
            device = x.device
            position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)  # (seq_len,1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(seq_len, d_model, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)  # (seq_len,1,d_model)
            return x + pe


# Transformer Decoder for Captioning
class TransformerDecoder(nn.Module):
    """
    Transformer Decoder.
    """

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, max_len=50, dropout=0.1):
        """
        :param vocab_size: size of vocabulary
        :param d_model: embedding size
        :param nhead: number of attention heads
        :param num_layers: number of decoder layers
        :param dim_feedforward: feedforward dim
        :param max_len: max caption length
        :param dropout: dropout
        """
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoded_captions, encoder_out, caption_lengths=None, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Forward propagation for training (teacher forcing).

        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, d_model)
        :param caption_lengths: caption lengths (unused for Transformer, but kept for compatibility)
        :param tgt_mask: causal mask (generated outside if needed)
        :param tgt_key_padding_mask: padding mask (generated outside if needed)
        :return: predictions (batch, seq_len, vocab_size), encoded_captions, decode_lengths, None, None
        """
        batch_size = encoded_captions.size(0)
        seq_len = encoded_captions.size(1)

        # Embed and add pos encoding
        embeddings = self.embedding(encoded_captions) * math.sqrt(self.d_model)
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch, d_model)
        embeddings = self.dropout(self.pos_encoding(embeddings))  # positional encoding handles dynamic lengths

        # If no mask provided, generate causal mask
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Padding mask from lengths if provided
        if caption_lengths is not None and tgt_key_padding_mask is None:
            # Create padding mask: (batch, seq_len)
            tgt_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            for i, length in enumerate(caption_lengths.squeeze(1).tolist()):
                tgt_key_padding_mask[i, length:] = True

        # Decoder forward: transformer expects (tgt, memory) with tgt shape (seq_len, batch, d_model) and memory (mem_len, batch, d_model)
        decoder_out = self.transformer_decoder(embeddings, encoder_out.transpose(0, 1), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        decoder_out = decoder_out.transpose(0, 1)  # Back to (batch, seq_len, d_model)

        predictions = self.fc(self.dropout(decoder_out))  # (batch, seq_len, vocab_size)

        # Return predictions first for easy use in training loop
        return predictions, encoded_captions, [seq_len - 1] * batch_size, None, None


# Full Model (for compatibility)
class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # accept masks and pass them through to the decoder; return predictions tensor
    def forward(self, images, captions, lengths=None, tgt_mask=None, tgt_key_padding_mask=None):
        features = self.encoder(images)  # (batch, num_pixels, d_model)
        preds, _, _, _, _ = self.decoder(captions, features, lengths, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return preds

    def caption(self, images, **kwargs):
        # For inference; use beam search from eval.py or caption.py if you implement it
        pass
