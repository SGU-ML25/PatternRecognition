"""SimpleBaseline (Xiao et al., ECCV 2018): ResNet backbone + 3 lớp deconv -> 17 heatmap."""
import torch
import torch.nn as nn
import torchvision

from pose_utils import NUM_KPTS

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class PoseNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, deconv_ch=256):
        super().__init__()
        fn, weights, out_ch = BACKBONES[backbone]
        net = fn(weights=weights if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # bỏ avgpool + fc, stride 32
        layers, ch = [], out_ch
        for _ in range(3):  # 8x8 -> 16 -> 32 -> 64
            layers += [nn.ConvTranspose2d(ch, deconv_ch, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(deconv_ch), nn.ReLU(inplace=True)]
            ch = deconv_ch
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(deconv_ch, NUM_KPTS, 1)
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.deconv(self.backbone(x)))


if __name__ == "__main__":
    for name in BACKBONES:
        m = PoseNet(name, pretrained=False)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(name, f"{n:.1f}M params", tuple(m(torch.zeros(1, 3, 256, 256)).shape))
