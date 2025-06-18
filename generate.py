from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# Hyperparameters
block_size = 64
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Transformer Model Definitions (copied from your script) ===

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        output = weights @ v
        return output

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.layer_norm_final = nn.LayerNorm(n_embd)
        self.language_model_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.language_model_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if idx_next.item() == 46:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# === Load model and vocab at startup ===
@app.on_event("startup")
def load_model():
    global model, vocab, decode
    with open("saved_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = torch.load("model.pth", map_location=device)
    model.eval()

    def decode_fn(ids):
        tokens = b"".join(vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    decode = decode_fn


# === Endpoint ===
@app.post("/generate")
async def generate_text(context: str = Form(...)):
    try:
        # Tokenize input (naively using vocab)
        input_ids = []
        for c in context.encode("utf-8"):
            for i, token in enumerate(vocab):
                if token == bytes([c]):
                    input_ids.append(i)
                    break
        if not input_ids:
            return JSONResponse(status_code=400, content={"error": "Invalid input context"})

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = model.generate(idx, max_new_tokens=500)[0].tolist()
        raw_text = decode(output_ids)

        # Optional filtering
        filtered = ""
        for char in reversed(raw_text):
            if char != '\n' and (
                ('A' <= char <= 'Z') or
                ('a' <= char <= 'z') or
                ('1' <= char <= '9')
            ):
                filtered = char + filtered

        return {"generated_text": filtered}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("WEBSITES_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)