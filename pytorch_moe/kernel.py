@triton.jit
def _fused_moe_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr,
                      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
                      # strides...; N, K, EM, num_valid_tokens as runtime i32
                      EM: tl.constexpr-ish runtime scalar, N, K, num_valid_tokens,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                      MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, ...):
    pid        = tl.program_id(axis=0)
    # group-major swizzle over (EM_pad/BM) × (N/BN) tiles → L2 reuse of B
    num_pid_m  = tl.cdiv(EM, BLOCK_SIZE_M)          # EM = len(sorted_token_ids)
    num_pid_n  = tl.cdiv(N,  BLOCK_SIZE_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # MoE-ness, part 1: which expert + which token rows (tensor lookups, no branches)
    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr): return  # early exit
    expert_id = tl.load(expert_ids_ptr + pid_m)                       # scalar load
    offs_token = tl.load(sorted_token_ids_ptr + pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    token_mask = offs_token < num_valid_tokens                        # kills padding slots

    # A gathers rows of x via sorted ids (//top_k de-flattens the k-repeats)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    # B slices THIS expert's weight matrix — the only expert dependence
    b_ptrs = b_ptr + expert_id * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):            # the K-loop Triton will pipeline
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b, accumulator)           # → tensor cores (§4, stage 3)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # MoE-ness, part 2: routed-weight multiply + scatter store to sorted slots
    if MUL_ROUTED_WEIGHT: accumulator *= tl.load(topk_weights_ptr + offs_token, mask=token_mask)[..., None]
    c_ptrs = c_ptr + offs_token_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c.dtype), mask=token_mask[:, None])
