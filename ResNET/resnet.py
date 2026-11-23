"""ResNet (He et al., 2015, https://arxiv.org/abs/1512.03385) - modele do CIFAR-10."""
import torch
from torch import nn
import torch.nn.functional as F


class ShortcutA(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.stride, self.pad = stride, out_ch - in_ch

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad:
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x


class Block(nn.Module):
    """y = ReLU(F(x) + x), F = conv-BN-ReLU-conv-BN. residual=False - to samo bez skrotu (plain)."""

    def __init__(self, in_ch, out_ch, stride=1, residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = ShortcutA(in_ch, out_ch, stride) if residual else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """6n+2 warstw: conv -> 3 etapy po n blokow (16/32/64 filtry) -> avg pool -> FC. ResNet-56: n=9."""

    def __init__(self, n=9, residual=True, widths=(16, 32, 64), classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(widths[0])
        blocks, in_ch = [], widths[0]
        for i, w in enumerate(widths):
            for j in range(n):
                stride = 2 if i > 0 and j == 0 else 1
                blocks.append(Block(in_ch, w, stride, residual))
                in_ch = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_ch, classes)
        self.depth = 6 * n + 2

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def response_layers(self):
        return [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]


if __name__ == "__main__":
    for residual in (True, False):
        net = ResNet(9, residual)
        out = net(torch.randn(2, 3, 32, 32))
        name = "resnet" if residual else "plain"
        print(name, "depth", net.depth, "params", sum(p.numel() for p in net.parameters()), "out", tuple(out.shape))
