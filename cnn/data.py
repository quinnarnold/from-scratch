import numpy as np
import os
import pickle
import tarfile
import urllib.request


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = os.path.join(os.path.dirname(__file__), "cifar-10-batches-py")


def download_cifar10(data_dir=None):
    """Download and extract CIFAR-10 if not already present."""
    if data_dir is None:
        data_dir = CIFAR10_DIR
    if os.path.exists(data_dir):
        return data_dir

    tar_path = data_dir + ".tar.gz"
    print(f"Downloading CIFAR-10 to {tar_path}...")
    urllib.request.urlretrieve(CIFAR10_URL, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(os.path.dirname(data_dir))
    os.remove(tar_path)
    return data_dir


def load_cifar10(data_dir=None):
    """Load CIFAR-10 dataset, downloading if needed.

    Returns (x_train, y_train, x_test, y_test) as numpy arrays.
    Images are float32 in [0, 1] with shape (N, 3, 32, 32).
    Labels are int64 with shape (N,).

    Note: pickle is required here because CIFAR-10's official distribution
    uses Python pickle format. The data is downloaded from the canonical
    University of Toronto source.
    """
    data_dir = download_cifar10(data_dir)

    x_train, y_train = [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f"data_batch_{i}")
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x_train.append(batch[b"data"])
        y_train.append(batch[b"labels"])

    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train).astype(np.int64)

    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x_test = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.array(batch[b"labels"], dtype=np.int64)

    return x_train, y_train, x_test, y_test


def normalize(x, mean=None, std=None):
    """Per-channel normalization. Computes stats from x if not provided."""
    if mean is None:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
    if std is None:
        std = x.std(axis=(0, 2, 3), keepdims=True)
    return (x - mean) / std, mean, std


def random_horizontal_flip(x, p=0.5):
    """Randomly flip images horizontally. x shape: (N, C, H, W)."""
    mask = np.random.rand(x.shape[0]) < p
    out = x.copy()
    out[mask] = out[mask, :, :, ::-1]
    return out


def random_crop(x, padding=4):
    """Random crop with zero-padding. x shape: (N, C, H, W)."""
    N, C, H, W = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.empty_like(x)
    for i in range(N):
        top = np.random.randint(0, 2 * padding + 1)
        left = np.random.randint(0, 2 * padding + 1)
        out[i] = padded[i, :, top:top + H, left:left + W]
    return out


class DataLoader:
    """Mini-batch iterator with optional shuffling and augmentation."""

    def __init__(self, x, y, batch_size=64, shuffle=True, augment=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

    def __iter__(self):
        n = len(self.x)
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)

        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            x_batch = self.x[batch_idx]

            if self.augment:
                x_batch = random_horizontal_flip(x_batch)
                x_batch = random_crop(x_batch)

            yield x_batch, self.y[batch_idx]

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size
