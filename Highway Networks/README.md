# Highway Networks on CIFAR-10

PyTorch implementation of **Highway Networks** (Srivastava, Greff, Schmidhuber, 2015, [arXiv:1505.00387](https://arxiv.org/abs/1505.00387)), trained on CIFAR-10.

## The idea

A highway layer lets the network choose, for each unit, whether to transform its input or pass it through unchanged:

```
y = H(x) · T(x) + x · (1 − T(x))
H(x) = ReLU(conv(x))            # transformation
T(x) = sigmoid(conv(x) + b_T)   # transform gate, values in (0, 1)
```

The gate bias `b_T` starts out negative (−2), so at the start of training every layer mostly copies its input. This lets gradients flow through very deep networks, which plain networks of the same depth struggle with.

## Files

| File | Contents |
|---|---|
| `highway.py` | `HighwayConv`, `PlainConv` (baseline without gates), `ConvNet` (full network) |
| `train.py` | training and evaluation |
| `plots.py` | learning curves, depth comparison (Fig. 1), gate analysis (Fig. 2) |

## Architecture

```
for each stage (widths 32, 64, 128):
    plain conv 3x3 (changes the number of channels) + ReLU
    `blocks` x HighwayConv (or PlainConv)
    MaxPool 2x2 (except after the last stage)
global average pooling -> Linear -> 10 classes
```

As in the paper (Section 2.1), plain layers change the number of channels and the image size, because the highway formula needs `x`, `H(x)` and `T(x)` to be the same shape. Weights use He initialization. With the default `blocks=6` the network has 22 layers and ~2.4M parameters.

The plain network is made √2 ≈ 1.41x wider (45, 91, 181 channels), so it has about as many parameters as the highway network, because it has no gate convolution `T`. The paper does the same thing with 71 vs 50 units per layer.

## Usage

```bash
pip install torch torchvision matplotlib numpy

python train.py                        # highway network, default settings
python train.py model=plain            # the same network without gates
python train.py epochs=30 blocks=4     # change any CONFIG value with key=value
python train.py depths=[1,2,4,8]       # plain vs highway at several depths (Fig. 1)
```

All settings and their defaults are in `CONFIG` at the top of `train.py`. CIFAR-10 downloads automatically to `./data`.

### Google Colab

Upload the folder to Google Drive. Set the runtime type to GPU, then run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/MyDrive/Highway Networks"
!python train.py
```

## Output

Each run is saved in `runs/<model>_L<depth>/`:

- `model.pt`: trained weights
- `history.json`: config and per-epoch loss and accuracy
- `curves.png`: loss and accuracy curves
- `gates.png` (highway only): gate biases, mean gate activity, gates and block outputs for one image (like Fig. 2)

A depth sweep also saves `runs/depth_comparison.png`, which shows the training loss of plain and highway networks at each depth (like Fig. 1).

## Training setup

SGD with momentum 0.9, lr 0.05 multiplied by 0.95 after every epoch, weight decay 5e-4, batch size 128, 60 epochs. Augmentation is random crop with 4px padding plus horizontal flip.

Differences from the paper: Fig. 1 and Fig. 2 in the paper use fully connected networks on MNIST and CIFAR-100. Here the same experiments are run with the convolutional network on CIFAR-10. The paper's CIFAR-10 results (Table 1) are 89–92% test accuracy.
