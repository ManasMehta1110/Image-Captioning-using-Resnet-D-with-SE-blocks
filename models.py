# models.py
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Helper functions
# ======================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=dilation, groups=groups,
        bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride,
        bias=False
    )

# ======================================================
# Squeeze-and-Excitation Block
# ======================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
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

# ======================================================
# SE-ResNet-D Bottleneck
# ======================================================
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,
                 norm_layer=None, se_ratio=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, se_ratio)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# ======================================================
# ResNet-50-D + SE Encoder Backbone
# ======================================================
class ResNetDSE(nn.Module):
    def __init__(self, block, layers, norm_layer=None, se_ratio=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64

        self.conv1 = conv3x3(3, 32, stride=2)
        self.bn1 = norm_layer(32)
        self.conv2 = conv3x3(32, 32)
        self.bn2 = norm_layer(32)
        self.conv3 = conv3x3(32, 64)
        self.bn3 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], se_ratio=se_ratio)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, se_ratio=se_ratio)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, se_ratio=se_ratio)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, se_ratio=se_ratio)

    def _make_layer(self, block, planes, blocks, stride=1, se_ratio=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, se_ratio=se_ratio)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, se_ratio=se_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

# ======================================================
# Encoder Wrapper
# ======================================================
class Encoder(nn.Module):
    def __init__(self, d_model=512, encoded_image_size=7, load_imagenet=False):
        super().__init__()

        self.resnet = ResNetDSE(SEBottleneck, [3, 4, 6, 3])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fc = nn.Linear(2048, d_model)

        if load_imagenet:
            self._load_imagenet_weights()

        self.fine_tune(False)

    def forward(self, images):
        x = self.resnet(images)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return self.fc(x)

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for layer in [self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for p in layer.parameters():
                p.requires_grad = fine_tune

    def _load_imagenet_weights(self):
        print("=> Loading ImageNet pretrained weights")
        imagenet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        pretrained = imagenet.state_dict()
        model = self.resnet.state_dict()

        matched = 0
        for k in model:
            if k in pretrained and model[k].shape == pretrained[k].shape:
                model[k] = pretrained[k]
                matched += 1

        self.resnet.load_state_dict(model)
        print(f"=> Loaded {matched} layers")

# ======================================================
# Attention Module
# ======================================================
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        weighted = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return weighted, alpha

# ======================================================
# LSTM Decoder with Attention
# ======================================================
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim,
                 vocab_size, encoder_dim=512, dropout=0.5):
        super().__init__()

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), self.fc.out_features).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum(l > t for l in decode_lengths)
            context, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], context], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, None, sort_ind
