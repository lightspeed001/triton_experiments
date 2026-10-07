# triton_experiments
## Triton Experimental Works


### Triton Installation & Setup

> **Install Triton**

```sh
pip install triton

```

> **Verify installation**

```python
import triton
print(triton.__version__)

```

> **Development Tips**

- Debugging: Use `TRITON_DEBUG=1` environment varible
- Visualization: Use `triton.tools.dissamble` to see generated PTX
-  Tuning: Use `triton.autotune` to find optimal block sizes

```python
# Example of autotuning
@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE': 32}),
    triton.Config({'BLOCK_SIZE': 64}),
    triton.Config({'BLOCK_SIZE': 128}),
    triton.Config({'BLOCK_SIZE': 256}),
], key=['n_elements'])
@triton.jit
def autotuned_kernel(..., BLOCK_SIZE: tl.constexpr):

    ...
```

### When to Use Triton :bulb:

- Custom operations not in PyTorch/TensorFlow
- Fused operations (combine multiple ops into one kernel)
- Memory bound operations (optimise memory access patterns)
- Compute-bound operatipns (optimize arithmetic intensity)
- Hardware-specific optimizations (tensor cores, etc.)

__When NOT to Use Triton__

- Simple operations already optimized in PyTorch
- Small tensors (overhead of jernel launch)
- Operations that don't benefit from custom implemention
- When you need portability across non-NVIDIA GPUs.
