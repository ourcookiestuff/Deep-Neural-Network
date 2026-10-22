"""Highway Networks (Srivastava et al., 2015, https://arxiv.org/abs/1505.00387) - modele do CIFAR-10."""
import torch
from torch import nn
import torch.nn.functional as F


class HighwayConv(nn.Module):
    """y = H(x) * T(x) + x * (1 - T(x)),  H = ReLU(conv),  T = sigmoid(conv + b_T)."""

    def __init__(self, channels, gate_bias=-2.0):
        super().__init__()
        self.H = nn.Conv2d(channels, channels, 3, padding=1)
        self.T = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.kaiming_normal_(self.H.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.T.weight, nonlinearity="relu")
        nn.init.zeros_(self.H.bias)
        # ujemny bias bramki -> na starcie warstwa prawie kopiuje wejscie (sekcja 2.2)
        nn.init.constant_(self.T.bias, gate_bias)

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        return F.relu(self.H(x)) * t + x * (1 - t)


class PlainConv(nn.Module):
    """y = ReLU(conv(x)) - baseline bez bramek."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.relu(self.conv(x))


class ConvNet(nn.Module):
    """Dla kazdego stage'a: zwykla konwolucja (zmiana liczby kanalow) -> `blocks` blokow highway/plain -> MaxPool.
    Na koncu global avg pool + Linear."""

    def __init__(self, model="highway", widths=(32, 64, 128), blocks=6, gate_bias=-2.0, classes=10):
        super().__init__()
        layers, in_ch = [], 3
        for i, w in enumerate(widths):
            stem = nn.Conv2d(in_ch, w, 3, padding=1)
            nn.init.kaiming_normal_(stem.weight, nonlinearity="relu")
            nn.init.zeros_(stem.bias)
            layers += [stem, nn.ReLU()]
            layers += [HighwayConv(w, gate_bias) if model == "highway" else PlainConv(w) for _ in range(blocks)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            in_ch = w
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, classes)
        self.depth = len(widths) * (blocks + 1) + 1  # warstwy z wagami

    def forward(self, x):
        x = self.features(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def highway_layers(self):
        return [m for m in self.features if isinstance(m, HighwayConv)]


if __name__ == "__main__":
    for model in ("highway", "plain"):
        net = ConvNet(model)
        out = net(torch.randn(2, 3, 32, 32))
        print(model, "depth", net.depth, "params", sum(p.numel() for p in net.parameters()), "out", tuple(out.shape))
