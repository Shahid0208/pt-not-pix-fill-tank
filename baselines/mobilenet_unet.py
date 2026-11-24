import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MobileNetUNet(nn.Module):
    """
    R4: CNN Baseline for ACCV.
    Uses MobileNetV3-Small backbone for an 'Edge' comparison.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=True).features
        self.enc1 = backbone[0:2]   # 16
        self.enc2 = backbone[2:4]   # 24
        self.enc3 = backbone[4:9]   # 40
        self.enc4 = backbone[9:12]  # 48
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Conv2d(48, 24, 3, 1, 1)
        self.dec2 = nn.Conv2d(24, num_classes, 1)

    def forward(self, x):
        # x: [B, 3, H, W]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d = self.up(e4)
        d = self.dec1(d) # Simplified skip-free decoder for baseline
        d = self.up(d)
        d = self.up(d)
        out = self.dec2(d)
        return out

if __name__ == "__main__":
    model = MobileNetUNet()
    dummy = torch.randn(1, 3, 128, 128)
    print(f"CNN Output shape: {model(dummy).shape}")


class MobileNetUNetSkip(nn.Module):
    """
    CNN baseline used for the synthetic segmentation runs (segmentation/): the same MobileNetV3-Small encoder
    with a U-Net decoder (skip connections at strides 2, 4, 8 and 16) and full input resolution at the output.
    MobileNetUNet above does not run: its enc4 gives 96 channels but dec1 expects 48.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights
        f = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None).features
        self.enc = nn.ModuleList([f[0], f[1], f[2:4], f[4:9], f[9:12]])   # 16@/2, 16@/4, 24@/8, 48@/16, 96@/32
        chans = [16, 16, 24, 48, 96]

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True))
        self.dec = nn.ModuleList([block(chans[i + 1] + chans[i], chans[i]) for i in range(3, -1, -1)])
        self.head = nn.Sequential(block(chans[0], 16), nn.Conv2d(16, num_classes, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()
        for d in self.dec:
            s = skips.pop()
            x = nn.functional.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], 1))
        x = nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(x)
