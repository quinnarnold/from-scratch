import math
import backend as B


def im2col(x, kH, kW, stride=1, padding=0):
    """Reshape input patches into columns for efficient convolution.

    Args:
        x: input array of shape (N, C, H, W)
        kH, kW: kernel height and width
        stride: stride of the convolution
        padding: zero-padding added to both sides

    Returns:
        cols: array of shape (N * out_h * out_w, C * kH * kW)
        out_h, out_w: output spatial dimensions
    """
    N, C, H, W = x.shape
    if padding > 0:
        x = B.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_pad, W_pad = x.shape[2], x.shape[3]
    out_h = (H_pad - kH) // stride + 1
    out_w = (W_pad - kW) // stride + 1

    cols = B.zeros((N, C, kH, kW, out_h, out_w))
    for i in range(kH):
        i_max = i + stride * out_h
        for j in range(kW):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    cols = B.transpose(cols, (0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im(dcols, x_shape, kH, kW, stride=1, padding=0):
    """Inverse of im2col -- scatter column gradients back to input shape.

    Args:
        dcols: gradient array of shape (N * out_h * out_w, C * kH * kW)
        x_shape: original input shape (N, C, H, W)
        kH, kW: kernel height and width
        stride: stride of the convolution
        padding: zero-padding that was applied

    Returns:
        dx: gradient w.r.t. input of shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    out_h = (H_pad - kH) // stride + 1
    out_w = (W_pad - kW) // stride + 1

    dcols_reshaped = B.transpose(
        dcols.reshape(N, out_h, out_w, C, kH, kW), (0, 3, 4, 5, 1, 2)
    )
    dx_padded = B.zeros((N, C, H_pad, W_pad))

    for i in range(kH):
        i_max = i + stride * out_h
        for j in range(kW):
            j_max = j + stride * out_w
            dx_padded[:, :, i:i_max:stride, j:j_max:stride] += dcols_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return dx_padded[:, :, padding:-padding, padding:-padding]
    return dx_padded


def softmax(x):
    """Numerically stable softmax over last axis."""
    shifted = x - B.amax(x, axis=-1, keepdims=True)
    exp_x = B.exp(shifted)
    return exp_x / B.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss with built-in softmax.

    Args:
        logits: raw scores of shape (N, C)
        targets: integer class labels of shape (N,)

    Returns:
        loss: scalar average loss
        dlogits: gradient w.r.t. logits of shape (N, C)
    """
    N = logits.shape[0]
    probs = softmax(logits)
    idx = B.arange(N)
    log_probs = -B.log(probs[idx, targets] + 1e-12)
    loss = B.mean(log_probs)

    dlogits = B.copy(probs)
    dlogits[idx, targets] -= 1.0
    dlogits /= N
    return loss, dlogits


def kaiming_init(shape, fan_in):
    """Kaiming He initialization for ReLU networks."""
    std = math.sqrt(2.0 / fan_in)
    return B.randn(*shape) * std


def xavier_init(shape, fan_in, fan_out):
    """Xavier/Glorot initialization."""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return B.randn(*shape) * std
