"""Wykresy: krzywe uczenia, porownanie glebokosci (Rys. 1), analiza bramek (Rys. 2)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"zapisano {path}")


def plot_history(hist, path, title=""):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.semilogy(ep, hist["train_loss"], label="train")
    a.semilogy(ep, hist["test_loss"], label="test")
    a.set(xlabel="epoka", ylabel="cross-entropy", title="Loss")
    b.plot(ep, hist["train_acc"], label="train")
    b.plot(ep, hist["test_acc"], label="test")
    b.set(xlabel="epoka", ylabel="accuracy", title="Accuracy")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    save(fig, path)


def plot_depth_comparison(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, model in zip(axes, ("plain", "highway")):
        for (m, depth), hist in sorted(histories.items(), key=lambda kv: kv[0][1]):
            if m == model:
                ax.semilogy(np.arange(1, len(hist["train_loss"]) + 1), hist["train_loss"], label=f"{depth} warstw")
        ax.set(xlabel="epoka", title=model)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("cross-entropy (trening)")
    save(fig, path)


def to_matrix(rows):
    mat = np.full((len(rows), max(len(r) for r in rows)), np.nan)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r.numpy()
    return mat


@torch.no_grad()
def plot_gates(net, loader, device, path, n_samples=1000):
    layers = net.highway_layers()
    gates, outs = [], []

    def hook(layer, inp, out):
        gates.append(torch.sigmoid(layer.T(inp[0])).mean((2, 3)).cpu())
        outs.append(out.mean((2, 3)).cpu())

    hooks = [l.register_forward_hook(hook) for l in layers]
    net.eval()
    seen = 0
    for x, _ in loader:
        net(x.to(device))
        seen += len(x)
        if seen >= n_samples:
            break
    for h in hooks:
        h.remove()

    L = len(layers)
    panels = [
        ("Bias bramek", [l.T.bias.detach().cpu() for l in layers], None),
        (f"Srednia aktywnosc bramek ({seen} obrazow)", [torch.cat(gates[i::L]).mean(0) for i in range(L)], None),
        ("Bramki - jeden obraz", [gates[i][0] for i in range(L)], (0, 1)),
        ("Wyjscia blokow - jeden obraz", [outs[i][0] for i in range(L)], None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, rows, lim) in zip(axes, panels):
        im = ax.imshow(to_matrix(rows), aspect="auto", cmap="coolwarm", interpolation="nearest",
                       vmin=lim and lim[0], vmax=lim and lim[1])
        ax.set(title=title, xlabel="kanal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("warstwa highway")
    save(fig, path)
