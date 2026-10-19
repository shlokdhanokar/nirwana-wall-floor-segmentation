"""WallFloorNet: a small encoder-decoder for 3-class (other/wall/floor) segmentation.

Encoder : torchvision MobileNetV3-Large *feature extractor* (optionally initialised
          from ImageNet-1k classification weights - a classifier, not a
          segmentation model). Not dilated, so stride 32 at the top.
Decoder : written for this project. A light-weight FPN:
            * context block on the stride-32 features: 1x1 projection gated by a
              global-average-pool "squeeze" (LR-ASPP idea) + an explicit
              global-context vector. Wall-vs-floor is largely a *layout*
              decision, so the top of the decoder needs to see the whole room.
            * top-down pathway fusing strides 16, 8 and 4 with depthwise-separable
              3x3 smoothing convs, keeping boundaries sharp at low cost.
            * 1x1 classifier at stride 4, bilinear upsampling to input size.
~3.3 M parameters total, ~1.3 GFLOPs at 512x512.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from .labels import NUM_CLASSES

# indices into mobilenet_v3_large().features whose outputs we tap, and their widths
_TAPS = {3: 24, 6: 40, 12: 112, 16: 960}  # strides 4, 8, 16, 32


def conv_bn_act(cin, cout, k=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SepConv(nn.Sequential):
    """Depthwise 3x3 + pointwise 1x1."""

    def __init__(self, ch):
        super().__init__(conv_bn_act(ch, ch, 3, groups=ch), conv_bn_act(ch, ch, 1))


class ContextBlock(nn.Module):
    def __init__(self, cin, ch):
        super().__init__()
        self.proj = conv_bn_act(cin, ch)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, ch, 1), nn.Sigmoid())
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, ch, 1), nn.ReLU(inplace=True))
        self.fuse = conv_bn_act(2 * ch, ch)

    def forward(self, x):
        local = self.proj(x) * self.gate(x)
        glob = self.glob(x).expand_as(local)
        return self.fuse(torch.cat([local, glob], 1))


class WallFloorNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, ch=96, pretrained_encoder=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained_encoder else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.context = ContextBlock(_TAPS[16], ch)
        self.lateral = nn.ModuleDict({str(i): conv_bn_act(_TAPS[i], ch) for i in (12, 6, 3)})
        self.smooth = nn.ModuleDict({str(i): SepConv(ch) for i in (12, 6, 3)})
        self.classifier = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(ch, num_classes, 1))

    def forward(self, x):
        size = x.shape[-2:]
        feats = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in _TAPS:
                feats[i] = x
        y = self.context(feats[16])
        for i in (12, 6, 3):
            lat = self.lateral[str(i)](feats[i])
            y = F.interpolate(y, size=lat.shape[-2:], mode="bilinear", align_corners=False)
            y = self.smooth[str(i)](y + lat)
        y = self.classifier(y)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WallFloorNet(pretrained_encoder=False, **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
