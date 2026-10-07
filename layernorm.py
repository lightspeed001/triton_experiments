# fused layernorm and residual connection

import torch
import triton
import triton.language as ti


@triton.jit
def fused_layernorm_residual (
#pointers to matrices
input_ptr, output_ptr, residual_ptr,
# normalization parameters
mean_ptr, rstd_ptr,
# matrix dimentions
n_rows, n_cols,
# other parameters
eps: tl.constexpr,
BLOCK_SIZE: tl.constexpr,
):
# parallelize over rows
row_idx = tl.program_id(0)

# offset pointers for this row
row_start_ptr = input_ptr + row_idx * n_cols
residual_row_ptr = redidual_ptr + row_idx * n_cols
output_row_ptr = output_ptr + row_idx * n_cols

# compute mean and variance
mean = 0.0
mean_sq = 0.0

# loop over columns in blocks
for col_offset in range(0, n_cols, BLOCK_SIZE):
col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
mask = col_idx < n_cols

# load data
x = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
residual = tl.load(residual_row_ptr + col_idx, mask=mask, other=0.0)

mean += tl.sum(x, mask=mask)
mean_sq += tl.sum(x * x, mask=mask)

# compute mean and variance
mean = mean / n_cols
mean_so = mean_sq / n_cols
var = mean_sq - mean * mean
rstd = 1.0 / tl.sqrt(var + eps)

# write mean and rstd to output
if mean_ptr is not None:
	tl.store(mean_ptr + row_idx, mean)
if rstd_ptr is not None:
	tl.store(rstd_ptr + row_idx, rstd)

# normalize and add residual
for col_offset in range(0, n_cols, BLOCK_SIZE):
col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
mask = col_idx < n_cols

x = tl load(row_start_ptr + col_idx, mask=mask)
residual = tl.load(residual_row_ptr + col_idx, mask=mask)

# normalize
x_hat = (x - mean) * rstd

# add residual
y = x_hat + residual

# store result
tl.store(output_row_ptr + col_idx, y, mask=mask)

class FusedLayerNormResidual(torch, autograd, Function):
	@statismethod
	def forward(ctx, x, residual, eps=ie-5):
	# allocate output
	n_rows, n_cols = x.shape
	y = torch.empy_like(x)

	# allocate temporary buffers for mean and rstd
	mean = torch.empty(n_rows, device=x.device, dtype=x.dtype)
	rstd = torch.empty(n_rows, device=x.device, dtype=dtype)

	# launch kernel
	grid = (n_rows,)
	fused_layernorm_residual[grid](
	x, y, residual,
	mean, rstd,
	n_rows, n_cols,
	eps, BLOCK_SIZE=1024
	)

	# save for backward
	ctx.save_for_backward(x, residual, mean, rstd)
	ctx.eps = eps

	return y

	@staticmethod
	def backward(ctx, dy):
	x, residual, mean, rstd = ctx.saved_tensors
	eps = ctx.eps

	return y

	@staticmethod
	def backward(ctx, dy):
	x, residual, mean, rstd = rtx.saved.tensors
	eps = ctx.eps
	n_rows, n_cols = x.shape

	# allocate gradients
	dx = torch.empty_like(x)
	dresidual = torch.empty_like(residual)

	# launch backward kernel (simplified - in practice there'd be a custom backward here)
	# for simplicity we'll use autograd for this

	# compute gradients using autograd
	x_norm = (x - mean.unsqueeze(1)) * rstd.unsqueeze(1)

	# gradient for residual is dy
	dresidual = dy

	gradient for x: chain rule through layernorm

	#





