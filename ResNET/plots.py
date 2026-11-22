"""Wykresy: krzywe uczenia, porownanie plain vs ResNet (Rys. 6), odpowiedzi warstw (Rys. 7)."""
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


def err(acc):
    return 100 * (1 - np.asarray(acc))


def plot_history(hist, path, title=""):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.semilogy(ep, hist["train_loss"], label="train")
    a.semilogy(ep, hist["test_loss"], label="test")
    a.set(xlabel="epoka", ylabel="cross-entropy", title="Loss")
    b.plot(ep, err(hist["train_acc"]), label="train")
    b.plot(ep, err(hist["test_acc"]), label="test")
    b.set(xlabel="epoka", ylabel="blad (%)", title="Blad", ylim=(0, 20))
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    save(fig, path)


def plot_depth_comparison(histories, path):
    """Rys. 6: przerywana = trening, ciagla = test."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, ("plain", "resnet")):
        runs = sorted((d, h) for (m, d), h in histories.items() if m == model)
        for (depth, hist), color in zip(runs, plt.cm.tab10.colors):
            ep = np.arange(1, len(hist["train_acc"]) + 1)
            ax.plot(ep, err(hist["train_acc"]), "--", color=color, lw=1)
            ax.plot(ep, err(hist["test_acc"]), color=color, lw=1.8, label=f"{model}-{depth}")
        ax.set(xlabel="epoka", title=model, ylim=(0, 20))
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("blad (%)")
    save(fig, path)


@torch.no_grad()
def layer_stds(net, loader, device, n_samples=1000):
    """Std odpowiedzi kazdej warstwy (wyjscie BN), srednia po kanalach."""
    layers = net.response_layers()
    acts = [[] for _ in layers]
    hooks = [l.register_forward_hook(lambda m, i, o, k=k: acts[k].append(o.cpu())) for k, l in enumerate(layers)]
    net.eval()
    seen = 0
    for x, _ in loader:
        net(x.to(device))
        seen += len(x)
        if seen >= n_samples:
            break
    for h in hooks:
        h.remove()
    return np.array([torch.cat(a).transpose(0, 1).flatten(1).std(1).mean().item() for a in acts])


def plot_responses(nets, loader, device, path):
    """Rys. 7: std warstw w kolejnosci i posortowane."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6))
    for ((model, depth), net), color in zip(sorted(nets.items()), plt.cm.tab10.colors):
        s = layer_stds(net, loader, device)
        style = "--" if model == "plain" else "-"
        a.plot(np.arange(1, len(s) + 1), s, style, color=color, label=f"{model}-{depth}")
        b.plot(np.arange(1, len(s) + 1), np.sort(s)[::-1], style, color=color, label=f"{model}-{depth}")
    a.set(xlabel="numer warstwy (kolejnosc w sieci)", ylabel="std")
    b.set(xlabel="numer warstwy (posortowane wg std)", ylabel="std")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, path)
