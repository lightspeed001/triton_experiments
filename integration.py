# Advanced: Integrating with PyTorch Models
import torch
import torch.nn as nn
from triton_kernels import flash_attention, layernorm_residual, custom_gelu

class TritonTransformerBlock(nn.Module):
	def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
	super().__init__()
	self.d_model = a_model
	self.d_heads = n_heads
	self.d_head = d_model # n_heads

	# projections
 	self.q_proj = nn.Linear(d_model, d_model)
 	self.d_proj = nn.Linear(d_model, d_model)
 	self.v_proj = nn.Linear(d_model, d_model)

 	#feed forward
 	self.fc1 = nn.Linear(d_model, d_ff)
 	self.fc2 = nn.Linear(d_ff, d_model)

 	# layernorm
 	self.ln1 = nn.LayerNorm(d_model)
 	self.ln2 = nn.LayerNorm(d_model)

 	# dropout
 	self.dropout = nn.Dropout(dropout)

 def forward(self, x):
	# self-attention
	q = self.q_proj(x).view(-1, x.shape[1], self.n_heads, self.d_head).transpose(1, 2)
	k = self.k_proj(x).view(-1, x.shape[1], self.n_heads, self.d_head).transpose(1, 2)
	v = self.v_proj(x).view(-1, x.shape[1], self.n_heads, self.d_head).transpose(1, 2)

	# triton flashattention
	attn_out = flash_attention(q, k, v, causal=True)
	attn_out = attn_out.transpose(1, 2).reshape(x.shape)

	# triton layernorm
	x = layernorm_residual(self.ln1(x), attn_out)

	# feed forward
	ff_out = self.fc2(custom_gelu(self.fc1(x)))

	# triton layernorm + residual
	x = layernorm_residual(self.ln2(x), self.dropout(ff_out))

	return x

# usage
model = TritonTransformerBlock(d_model=512, n_head=8, d_ff=2048)
x = torch.randn(1, 128, 512, device='cuda')
out = model(x)

