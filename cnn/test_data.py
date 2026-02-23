import numpy as np
import pytest
from data import DataLoader, normalize, random_horizontal_flip, random_crop


def test_normalize():
    np.random.seed(42)
    x = np.random.rand(100, 3, 8, 8).astype(np.float32)

    x_norm, mean, std = normalize(x)

    assert x_norm.shape == x.shape
    recomputed_mean = x_norm.mean(axis=(0, 2, 3))
    recomputed_std = x_norm.std(axis=(0, 2, 3))
    np.testing.assert_allclose(recomputed_mean, 0.0, atol=1e-5)
    np.testing.assert_allclose(recomputed_std, 1.0, atol=0.05)

    x_norm2, _, _ = normalize(x, mean=mean, std=std)
    np.testing.assert_array_equal(x_norm, x_norm2)


def test_random_horizontal_flip():
    np.random.seed(42)
    x = np.random.rand(200, 3, 8, 8).astype(np.float32)
    flipped = random_horizontal_flip(x, p=0.5)

    assert flipped.shape == x.shape
    n_flipped = 0
    for i in range(len(x)):
        if np.array_equal(flipped[i], x[i, :, :, ::-1]):
            n_flipped += 1
        else:
            assert np.array_equal(flipped[i], x[i])
    assert 60 < n_flipped < 140


def test_random_crop():
    np.random.seed(42)
    x = np.random.rand(10, 3, 8, 8).astype(np.float32)
    cropped = random_crop(x, padding=4)

    assert cropped.shape == x.shape
    assert not np.array_equal(cropped, x)


def test_dataloader_basic():
    np.random.seed(42)
    x = np.random.rand(100, 3, 8, 8).astype(np.float32)
    y = np.random.randint(0, 10, 100).astype(np.int64)

    loader = DataLoader(x, y, batch_size=32, shuffle=False, augment=False)
    assert len(loader) == 4

    batches = list(loader)
    assert len(batches) == 4
    assert batches[0][0].shape == (32, 3, 8, 8)
    assert batches[0][1].shape == (32,)
    assert batches[-1][0].shape == (4, 3, 8, 8)

    all_x = np.concatenate([b[0] for b in batches])
    np.testing.assert_array_equal(all_x, x)


def test_dataloader_shuffle():
    np.random.seed(42)
    x = np.arange(50).reshape(50, 1, 1, 1).astype(np.float32)
    y = np.arange(50).astype(np.int64)

    loader = DataLoader(x, y, batch_size=50, shuffle=True)
    batch_x, batch_y = next(iter(loader))

    assert not np.array_equal(batch_y, y)
    assert set(batch_y.tolist()) == set(y.tolist())


def test_dataloader_augment():
    np.random.seed(42)
    x = np.random.rand(16, 3, 8, 8).astype(np.float32)
    y = np.zeros(16, dtype=np.int64)

    loader = DataLoader(x, y, batch_size=16, shuffle=False, augment=True)
    batch_x, _ = next(iter(loader))

    assert batch_x.shape == x.shape
    assert not np.array_equal(batch_x, x)
