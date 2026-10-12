import torch
import triton
import triton.language as tl

@triton.jit
def softmax_cross_entropy_forward(
    logits_ptr, labels_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over rows
    row_idx = tl.program_id(0)
    
    # Offset pointers
    row_start_ptr = logits_ptr + row_idx * n_cols
    labels_row_ptr = labels_ptr + row_idx * n_cols
    
    # Load logits and labels
    logits = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    labels = tl.load(labels_row_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Compute max for numerical stability
    max_logit = tl.max(logits, axis=0)
    
    # Compute exp(logit - max) and sum
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    # Compute softmax
    softmax = exp_logits / sum_exp
    
    # Compute cross-entropy: -sum(labels * log(softmax))
    # For numerical stability: -sum(labels * (logits - max_logit - log(sum_exp)))
    log_softmax = logits - max_logit - tl.log(sum_exp)
    loss = -tl.sum(labels * log_softmax, axis=0)
    
    # Store output
    tl.store(output_ptr + row_idx, loss)

@triton.jit
def softmax_cross_entropy_backward(
    logits_ptr, labels_ptr, output_grad_ptr, input_grad_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over rows
    row_idx = tl.program_id(0)
    
    # Offset pointers
    row_start_ptr = logits_ptr + row_idx * n_cols
    labels_row_ptr = labels_ptr + row_idx * n_cols
    
    # Load logits, labels, and output gradient
    logits = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    labels = tl.load(labels_row_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    d_loss = tl.load(output_grad_ptr + row_idx)
    
    # Compute max for numerical stability
    max_logit = tl.max(logits, axis=0)
    
    # Compute exp(logit - max) and sum
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    # Compute softmax
    softmax = exp_logits / sum_exp
    
    # Gradient: d_loss * (softmax - labels)
    d_logits = d_loss * (softmax - labels)
    
    # Store input gradient
    tl.store(input_grad_ptr + row_start_ptr + tl.arange(0, BLOCK_SIZE), 
              d_logits, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

class SoftmaxCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        # Allocate output
        n_rows = logits.shape[0]
        loss = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)
        
        # Launch kernel
        grid = (n_rows,)
        softmax_cross_entropy_forward[grid](
            logits.data_ptr(), labels.data_ptr(), loss.data_ptr(),
            n_rows, logits.shape[1], BLOCK_SIZE=1024
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels)
        
        return loss.mean()
    
    @staticmethod
    def backward(ctx, d_loss):
        logits, labels = ctx.saved_tensors
        
        # Allocate input gradient
        d_logits = torch.empty_like(logits)
        
        # Scale d_loss by 1/n_rows (because we took mean in forward)
        d_loss_scaled = d_loss / logits.shape[0]
        
        # Launch backward kernel
        grid = (logits.shape[0],)
        softmax_cross_entropy_backward[grid](
            logits.data_ptr(), labels.data_ptr(),
            d_loss_scaled.data_ptr(), d_logits.data_ptr(),
            logits.shape[0], logits.shape[1], BLOCK_SIZE=1024
        )
        
        return d_logits, None

def softmax_cross_entropy(logits, labels):
    return SoftmaxCrossEntropyLoss.apply(logits, labels)

# Test and benchmark
if __name__ == "__main__":
    import time
    
    # Create test data
    batch_size = 1024
    n_classes = 1000
    
    logits = torch.randn(batch_size, n_classes, device='cuda')
    labels = torch.zeros(batch_size, n_classes, device='cuda')
    
    # Create one-hot labels
    targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
    labels.scatter_(1, targets.unsqueeze(1), 1.0)
    
    # Warmup
    for _ in range(10):
        loss = softmax_cross_entropy(logits, labels)
    
    # Benchmark Triton
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        loss = softmax_cross_entropy(logits, labels)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Triton Softmax CE: {elapsed / n_iters * 1000:.2f} ms")
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(n_iters):
        loss = torch.nn.functional.cross_entropy(logits, targets)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"PyTorch Softmax CE: {elapsed / n_iters * 1000:.2f} ms")
    
    # Verify correctness
    logits_test = torch.randn(100, 10, device='cuda')
    labels_test = torch.zeros(100, 10, device='cuda')
    targets_test = torch.randint(0, 10, (100,), device='cuda')
    labels_test.scatter_(1, targets_test.unsqueeze(1), 1.0)
    
    loss_triton = softmax_cross_entropy(logits_test, labels_test)
    loss_torch = torch.nn.functional.cross_entropy(logits_test, targets_test)
    
    print(f"Loss difference: {abs(loss_triton - loss_torch).item():.6f}")

