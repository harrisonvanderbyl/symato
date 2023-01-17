import math
import torch
from torch import nn
from torch.nn import functional as F

def new_gelu(x):
	""" GELU motivation: combine ReLU and Dropout `docs/gelu.md` được tính xấp xỉ theo công thức
	"""
	a = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
	return 0.5*x*(1.0 + torch.tanh(a))

class CasualSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0, "Số lượng embed không chia hết cho số lượng head"
		# Ánh xạ K,Q,V cho tất cả các head
		self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
		self.c_proj = nn.Linear(config.n_embd, config.n_embd)
		# Regularization
		if config.dropout > 0:
			self.attn_dropout = nn.Dropout(config.dropout)
			self.resd_dropout = nn.Dropout(config.dropout)
		# Mặt nạ nhân quả để đảm bảo attn chỉ áp dụng từ trái qua phải của chuỗi đầu vào
		n = config.block_size
		self.register_buffer("bias", torch.tril(torch.ones(n, n)).view(1,1,n,n))
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.dropout = config.dropout

	def forward(self, x):
		nh, ne = self.n_head, self.n_embd
		B, T, C = x.size(); assert C == ne # batch size, sequence length, embdedding dimensionality (n_embd)
		q, k, v = self.c_attn(x).split(ne, dim=2)

		hs = ne // nh # head_size = n_embd / n_head
		# (B, T, nh, hs) => (B, nh, T, hs)
		k = k.view(B, T, nh, hs).transpose(1, 2) 
		q = q.view(B, T, nh, hs).transpose(1, 2)
		v = v.view(B, T, nh, hs).transpose(1, 2)

		# (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.T) * (1.0 / math.sqrt(T))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if self.dropout > 0: att = self.attn_dropout(att)
        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        if self.dropout > 0: y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd) # tại sao lại 4x?
		self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
		if config.dropout > 0: self.dropout = nn.Dropout(config.dropout)
		self.dropout = config.dropout

	def forward(self, x):
		x = self.c_fc(x)
		x = new_gelu(x)
		x = self.c_proj(x)
		if self.dropout > 0: x = self.dropout(x)
		return x

class Block(nn.Module):
	def __init__(self, config):
		super.__inint__()
		self.ln_1 = nn.LayerNorm(config.n_embd)
		self.attn = CasualSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embd)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x

@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50257
	n_layer: int = 12
	n_head: int = 12
	n_embd: int = 768
	dropout: float = 0.1

class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config

		self.tfm = nn.ModuleDict(dict(
			tok_embd = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
			pos_embd = nn.Embedding(config.block_size, config.n_embd), # position embedding
			drop = nn.Dropout(config.dropout),
			blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			layernorm = nn.LayerNorm(config.n_embd),
		))

		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language model head
		# https://paperswithcode.com/method/weight-tying
		self.tfm.tok_emb.weight = self.lm_head.weight

		n_params = sum(p.numel() for p in self.parameters())
		print(">>> number of parameters: %.2fM" % (n_params / 1e6))

	def forward(self, idx, targets=None):
		_, t = idx.size()
		block_size = self.config.block_size
		assert t < block_size, f"Cannot forward sequence of length {t} with block size {block_size}"
		pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0) # shape (1, t)

		# Biến idx vectors đầu vào thành embedding vectors x
		x  = self.tfm.tok_embd(idx) # token embeddings of shape (b, t, n_embd)
		x += self.tfm.pos_embd(pos) # position embeddings of shape (1, t, n_embd)
		if self.config.dropout > 0: x = self.tfm.drop(x)
		for block in self.tfm.blocks: x = block(x)
		x = self.tfm.layernorm(x)

		if target is not None:
			logits = self.lm_head(x)
			targets_ = logits.view(-1, logits.size(-1))
			loss = F.cross_entropy(targets_, targets.view(-1), ignore_index=-1)
		else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
