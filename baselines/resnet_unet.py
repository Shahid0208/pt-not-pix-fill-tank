import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18UNet(nn.Module):
    """
    Segmentation with the keypoint model's backbone: ImageNet-pretrained ResNet-18 encoder with
    the same U-Net decoder as baselines/mobilenet_unet.MobileNetUNetSkip (skip connections at strides 2, 4, 8 and
    16, full input resolution at the output). Used with 256 px crops over two rim diameters, the keypoint
    model's crop and resolution, so that backbone and input resolution match the keypoint pipeline.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.enc = nn.ModuleList([nn.Sequential(r.conv1, r.bn1, r.relu),         # 64@/2
                                  nn.Sequential(r.maxpool, r.layer1),             # 64@/4
                                  r.layer2, r.layer3, r.layer4])                  # 128@/8, 256@/16, 512@/32
        chans = [64, 64, 128, 256, 512]

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True))
        self.dec = nn.ModuleList([block(chans[i + 1] + chans[i], chans[i]) for i in range(3, -1, -1)])
        self.head = nn.Sequential(block(chans[0], 32), nn.Conv2d(32, num_classes, 1))

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
