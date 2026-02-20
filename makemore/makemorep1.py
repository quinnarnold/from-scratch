import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()

stoi = {s: i + 1 for i, s in enumerate(sorted(set("".join(words))))}
stoi["."] = 0
itos = {i: s for i, s in stoi.items()}

W = torch.randn((27, 27), requires_grad=True)
for epoch in range(10):
    xs, ys = [], []
    for w in words:
        chars = ["."] + list(w) + ["."]
        for char1, char2 in zip(chars, chars[1:]):
            idx1 = stoi[char1]
            idx2 = stoi[char2]

            xs.append(idx1)
            ys.append(idx2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    log_tensor = logits.exp()
    probs = log_tensor / log_tensor.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(ys)), ys].log().mean()

    W.grad = None
    loss.backward()
    W.data += -100 * W.grad
    print(loss)
