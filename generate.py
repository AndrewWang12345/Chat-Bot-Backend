from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import bcrypt
import sqlalchemy
import databases
from fastapi import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED
from pydantic import BaseModel
from sqlalchemy import Table, Column, Integer, String, Text

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
    model = torch.load("model.pth", map_location=device,weights_only=False)
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

DATABASE_URL = "postgresql://postgres.hneswphqvkiovndybvyp:[password]@aws-0-ca-central-1.pooler.supabase.com:5432/postgres"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("username", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("password", sqlalchemy.String),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL.replace("+asyncpg", ""), echo=True
)
metadata.create_all(engine)

class User(BaseModel):
    username: str
    password: str

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/api/auth/register")
async def register_user(user: User):
    hashed_pw = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    query = users.insert().values(username=user.username, password=hashed_pw)
    try:
        await database.execute(query)
        return {"status": True, "user": {"username": user.username}}
    except Exception as e:
        return {"status": False, "error": str(e)}
@app.post("/api/auth/login")
async def login_user(user: User):
    #print(User.username,user.password)
    query = users.select().where(users.c.username == user.username)
    db_user = await database.fetch_one(query)

    if db_user is None:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="User not found")

    if not bcrypt.checkpw(user.password.encode(), db_user["password"].encode()):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid password")

    return {"status": True, "user": {"username": db_user["username"]}}

class UserOut(BaseModel):
    username: str

@app.get("/api/auth/allUsers/{user_id}", response_model=list[UserOut])
async def get_all_users(user_id: str):
    try:
        print(user_id)
        query = users.select().where(users.c.username != user_id)
        print(user_id)
        result = await database.fetch_all(query)
        print(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



chat_history = Table(
    "chat_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String, nullable=False),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
)

class ChatEntry(BaseModel):
    username: str
    question: str
    answer: str

@app.post("/api/chat/save")
async def save_chat_entry(entry: ChatEntry):
    query = chat_history.insert().values(
        username=entry.username,
        question=entry.question,
        answer=entry.answer
    )
    await database.execute(query)
    return {"status": "ok"}

class ChatMessage(BaseModel):
    sender: str  # 'user' or 'ai'
    message: str

@app.get("/api/chat/history/{username}", response_model=list[ChatMessage])
async def get_chat_history(username: str):
    query = chat_history.select().where(chat_history.c.username == username).order_by(chat_history.c.id)
    rows = await database.fetch_all(query)
    result = []
    for row in rows:
        # map your db fields to 'sender' and 'message'
        result.append({"sender": "user", "message": row["question"]})
        result.append({"sender": "ai", "message": row["answer"]})
    return result

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("WEBSITES_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)