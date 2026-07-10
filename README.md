## Running on AMD Instinct MI300X

All LLM inference runs on a single AMD Instinct MI300X (AMD Developer Cloud),
serving Gemma 3 4B through vLLM 0.23.0 on ROCm 7.2.4. No external LLM API is
used at any point.

### GPU

```
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
==========================================================================================================================
0       1     0x74b5,   21947  40.0°C      155.0W    NPS1, SPX, 0        139Mhz  900Mhz  0%   auto  750.0W  0%     0%
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================
```

![rocm-smi on the MI300X](evidence/rocm-smi.png)

### ROCm inference backend

Excerpts from [`evidence/vllm.log`](evidence/vllm.log):

```
[model.py:611]       Resolved architecture: Gemma3ForConditionalGeneration
[rocm.py:637]        Using Flash Attention backend for ViT model.
[rocm.py:583]        Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
                     Overriding with ROCM_ATTN out of potential backends:
                     ['ROCM_ATTN', 'ROCM_AITER_UNIFIED_ATTN', 'TRITON_ATTN'].
[activation.py:728]  [ROCm] PyTorch's native GELU with tanh approximation is unstable.
[gpu_worker.py:480]  Available KV cache memory: 162.84 GiB
[kv_cache_utils.py]  GPU KV cache size: 1,217,591 tokens
[kv_cache_utils.py]  Maximum concurrency for 8,192 tokens per request: 148.63x
[api_server.py:583]  Starting vLLM server on http://0.0.0.0:8000
INFO:                POST /v1/chat/completions HTTP/1.1" 200 OK
```

Model load: 8.58 GiB, 12.4 seconds. Weights downloaded in 8.0 seconds.

> vLLM logs `device_config=cuda` and "Capturing CUDA graphs" even on ROCm —
> HIP mirrors the CUDA API surface, so the naming persists. The `[rocm.py]`
> backend-selection lines and the 162 GiB KV cache confirm the AMD device.
```

Model load: 8.58 GiB, 12.4 seconds.

### Workload

_(to be filled: profiles generated, wall time, tokens/sec)_

### Reproduce

Full logs in [`evidence/`](evidence/). Setup: [`setup.md`](setup.md).