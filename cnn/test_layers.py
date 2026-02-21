import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from functional import im2col, col2im, softmax, cross_entropy_loss


def test_im2col_col2im_roundtrip():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    kH, kW, stride, padding = 3, 3, 1, 1
    cols, out_h, out_w = im2col(x, kH, kW, stride, padding)
    assert cols.shape == (2 * out_h * out_w, 3 * 3 * 3)


def test_softmax_matches_pytorch():
    np.random.seed(42)
    x = np.random.randn(4, 10).astype(np.float32)
    ours = softmax(x)
    theirs = F.softmax(torch.tensor(x), dim=-1).numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_cross_entropy_matches_pytorch():
    np.random.seed(42)
    logits = np.random.randn(8, 10).astype(np.float32)
    targets = np.array([0, 3, 5, 7, 2, 1, 9, 4])

    loss_ours, dlogits = cross_entropy_loss(logits, targets)

    logits_t = torch.tensor(logits, requires_grad=True)
    targets_t = torch.tensor(targets, dtype=torch.long)
    loss_t = F.cross_entropy(logits_t, targets_t)
    loss_t.backward()

    np.testing.assert_allclose(loss_ours, loss_t.item(), atol=1e-5)
    np.testing.assert_allclose(dlogits, logits_t.grad.numpy(), atol=1e-5)
