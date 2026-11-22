"""Trening ResNet (albo plain) na CIFAR-10 wg sekcji 4.2 artykulu.

python train.py                  # ResNet-56
python train.py model=plain      # plain-56
python train.py depths=[3,9]     # plain i resnet 20/56 + porownanie (Rys. 6)

Po przerwaniu ponowne uruchomienie wznawia trening z checkpointu.
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
from resnet import ResNet

CONFIG = dict(
    model="resnet",
    n=9,
    epochs=164,
    batch_size=128,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    milestones=[82, 123],  # = 32k i 48k iteracji z artykulu (koniec: 164 epoki = 64k)
    warmup_epochs=0,  # lr/10 na poczatku (artykul: ResNet-110)
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
    pin = torch.cuda.is_available()
    return (DataLoader(train, cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=pin, persistent_workers=True),
            DataLoader(test, 1000, num_workers=2, pin_memory=pin, persistent_workers=True))


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


def lr_at(cfg, epoch):
    if epoch < cfg["warmup_epochs"]:
        return cfg["lr"] * 0.1
    return cfg["lr"] * 0.1 ** sum(epoch >= m for m in cfg["milestones"])


def train(cfg, model, n, data, device):
    torch.manual_seed(0)
    train_dl, test_dl = data
    net = ResNet(n, residual=(model == "resnet")).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    name = f"{model}{net.depth}"
    out = Path(cfg["out"]) / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{name}: {sum(p.numel() for p in net.parameters()):,} parametrow, {device}")

    hist = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[], lr=[])
    ckpt_path, start = out / "checkpoint.pt", 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        hist, start = ckpt["hist"], ckpt["epoch"]
        print(f"wznawiam od epoki {start + 1}")

    for epoch in range(start, cfg["epochs"]):
        t0 = time.time()
        lr = lr_at(cfg, epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        loss_sum, correct, count = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        test_loss, test_acc = evaluate(net, test_dl, device)
        for k, v in zip(hist, (loss_sum / count, correct / count, test_loss, test_acc, lr)):
            hist[k].append(v)
        print(f"[{epoch + 1}/{cfg['epochs']}] lr {lr:g} | train loss {loss_sum / count:.4f} err {1 - correct / count:.2%} | "
              f"test loss {test_loss:.4f} err {1 - test_acc:.2%} | {time.time() - t0:.0f}s")
        if not math.isfinite(loss_sum):
            print("loss = NaN - trening sie rozbiegl")
            break
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), hist=hist, epoch=epoch + 1), ckpt_path)

    torch.save(net.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(dict(cfg=dict(cfg, model=model, n=n), **hist), indent=2))
    plots.plot_history(hist, out / "curves.png", f"{name} - najlepszy test err {1 - max(hist['test_acc']):.2%}")
    return hist, net


def main():
    cfg = parse_args(dict(CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    data = get_data(cfg)
    if cfg["depths"]:
        histories, nets = {}, {}
        for n in cfg["depths"]:
            for model in ("plain", "resnet"):
                hist, net = train(cfg, model, n, data, device)
                histories[(model, net.depth)], nets[(model, net.depth)] = hist, net
        plots.plot_depth_comparison(histories, Path(cfg["out"]) / "depth_comparison.png")
        plots.plot_responses(nets, data[1], device, Path(cfg["out"]) / "responses.png")
    else:
        hist, net = train(cfg, cfg["model"], cfg["n"], data, device)
        plots.plot_responses({(cfg["model"], net.depth): net}, data[1], device,
                             Path(cfg["out"]) / f"{cfg['model']}{net.depth}" / "responses.png")


if __name__ == "__main__":
    main()
