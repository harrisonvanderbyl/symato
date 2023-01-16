import math
import torch
from torch import nn

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

	def forward(self, x):
		B, T, C = x.size()