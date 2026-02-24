import torch
import torch.nn as nn
from torch.nn import functional as F

text = open("input.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}

encoder = lambda e: [stoi[c] for c in e]
decoder = lambda d: "".join([itos[c] for c in d])

data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]


def get_batch(split):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


max_iters = 5000
eval_iter = 1000
block_size = 8
batch_size = 4
n_embd = 32


@torch.no_grad()
def get_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out
    model.train()


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # Get Token Embedding
        pos_emb = self.position_embedding_table(torch.arange(T))  # Position
        x = tok_emb + pos_emb  # Final embed
        logits = self.lm_head(x)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        if targets is not None:
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_token=100):
        for _ in range(max_token):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramModel()
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
for batch in range(max_iters):
    if batch % eval_iter == 0:
        loss = get_loss()
        print(
            f"Iteration #{batch + eval_iter} Train Loss = {loss['train']:.4f} Val Loss = {loss['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
