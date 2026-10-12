import torch
import triton
import triton.language as tl

# Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@triton.jit
def gelu_forward(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load input
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Fast GELU approximation
    # Coefficients
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715

    # Compute GELU
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))

    # Store output
    tl.store(output_ptr + offsets, gelu, mask=mask)

@triton.jit
def gelu_backward(
    input_ptr, output_grad_ptr, input_grad_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load input and output gradient
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(output_grad_ptr + offsets, mask=mask, other=0.0)

    # Fast GELU derivative approximation
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715

    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_inner = tl.tanh(inner)

    # Derivative: 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh(inner)^2) * sqrt(2/pi) * (1 + 3 * coeff * x^2)
    term1 = 0.5 * (1.0 + tanh_inner)
    term2 = 0.5 * x * (1.0 - tanh_inner * tanh_inner) * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
    dgelu = term1 + term2

    # Compute input gradient
    dx = dy * dgelu

    # Store input gradient
    tl.store(input_grad_ptr + offsets, dx, mask=mask)

class CustomGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Allocate output
        y = torch.empty_like(x)

        # Launch kernel
        n_elements = x.numel()
        grid = triton.cdiv(n_elements, 1024)
        gelu_forward[grid](x.data_ptr(), y.data_ptr(), n_elements, BLOCK_SIZE=1024)

        # Save input for backward
        ctx.save_for_backward(x)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors

        # Allocate input gradient
        dx = torch.empty_like(x)

        # Launch backward kernel
        n_elements = x.numel()
        grid = triton.cdiv(n_elements, 1024)
        gelu_backward[grid](
            x.data_ptr(), dy.data_ptr(), dx.data_ptr(),
            n_elements, BLOCK_SIZE=1024
        )

        return dx

def custom_gelu(x):
    return CustomGELU.apply(x)

# Test and benchmark
if __name__ == "__main__":
    import time

    # Create test data
    x = torch.randn(1024, 1024, device='cuda')

    # Warmup
    for _ in range(10):
        y = custom_gelu(x)

    # Benchmark Triton GELU
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        y = custom_gelu(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Triton GELU: {elapsed / n_iters * 1000:.2f} ms")

    # Benchmark PyTorch GELU
    start = time.time()
    for _ in range(n_iters):
        y = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"PyTorch GELU: {elapsed / n_iters * 1000:.2f} ms")

    # Verify correctness
    x_test = torch.randn(100, device='cuda')
    y_triton = custom_gelu(x_test)
    y_torch = torch.nn.functional.gelu(x_test)

    print(f"Max difference: {torch.max(torch.abs(y_triton - y_torch)).item():.6f}")
