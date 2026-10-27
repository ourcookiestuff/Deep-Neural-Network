"""Trening Highway Network (albo plain) na CIFAR-10.

python train.py                            # highway z domyslnym CONFIG
python train.py model=plain epochs=30      # zmiana parametrow: klucz=wartosc
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

import plots
from highway import ConvNet

CONFIG = dict(
    model="highway",
    widths=[32, 64, 128],
    blocks=6,
    gate_bias=-2.0,
    epochs=60,
    batch_size=128,
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4,
    lr_decay=0.95,
    depths=None,
    out="runs",
)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def parse_args(cfg):
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        if k not in cfg:
            sys.exit(f"nieznana opcja {k}")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def get_data(cfg):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] + norm)
    train = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10("data", train=False, download=True, transform=T.Compose(norm))
    return (DataLoader(train, cfg["batch_size"], shuffle=True, num_workers=2),
            DataLoader(test, 1000, num_workers=2))


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss / n, correct / n


def train(cfg, model, blocks, data, device):
    torch.manual_seed(0)
    train_dl, test_dl = data
    widths = [round(w * math.sqrt(2)) for w in cfg["widths"]] if model == "plain" else cfg["widths"]
    net = ConvNet(model, widths, blocks, cfg["gate_bias"]).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, cfg["lr_decay"])
    name = f"{model}_L{net.depth}"
    out = Path(cfg["out"]) / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{name}: {sum(p.numel() for p in net.parameters()):,} parametrow, {device}")

    hist = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        net.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        test_loss, test_acc = evaluate(net, test_dl, device)
        for k, v in zip(hist, (loss_sum / n, correct / n, test_loss, test_acc)):
            hist[k].append(v)
        print(f"[{epoch}/{cfg['epochs']}] train loss {loss_sum / n:.4f} acc {correct / n:.3f} | "
              f"test loss {test_loss:.4f} acc {test_acc:.3f} | {time.time() - t0:.0f}s")
        if not math.isfinite(loss_sum):
            print("loss = NaN - trening sie rozbiegl")
            break

    torch.save(net.state_dict(), out / "model.pt")
    run_cfg = dict(cfg, model=model, blocks=blocks, widths=widths)
    (out / "history.json").write_text(json.dumps(dict(cfg=run_cfg, **hist), indent=2))
    plots.plot_history(hist, out / "curves.png", name)
    if model == "highway" and math.isfinite(loss_sum):
        plots.plot_gates(net, test_dl, device, out / "gates.png")
    return hist, net


def main():
    cfg = parse_args(dict(CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_data(cfg)
    if cfg["depths"]:
        histories = {}
        for blocks in cfg["depths"]:
            for model in ("plain", "highway"):
                hist, net = train(cfg, model, blocks, data, device)
                histories[(model, net.depth)] = hist
        plots.plot_depth_comparison(histories, Path(cfg["out"]) / "depth_comparison.png")
    else:
        train(cfg, cfg["model"], cfg["blocks"], data, device)


if __name__ == "__main__":
    main()
