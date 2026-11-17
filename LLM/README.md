# Block 1: LLM (sessions 1 to 14)

![One model that grows through the blocks](../docs/img/model-growth.svg)

Product: a chat model of 63.9 M parameters that thinks before it answers.
Design: Qwen3 at a small size (byte-level BPE, RMSNorm, RoPE, grouped-query
attention with QK-norm, SwiGLU, tied embedding). The full lesson table and the data
are in [`docs/specification.md`](../docs/specification.md), section S-5.

## Week 1 (milestone M1a)

| Session | Lesson | You write | Check |
|---------|--------|-----------|-------|
| 1 | [L01 BPE tokenizer](L01_tokenizer/README.md) | pre-tokens, BPE training, encode, decode | `ctest --preset clang -L L01` |
| 2 | [L02 Token data](L02_token_data/README.md) | token files, pre-training samples, batches | `-L L02` |
| 3 | [L03 RMSNorm and SwiGLU](L03_rmsnorm_swiglu/README.md) | the norm and the feed-forward layer | `-L L03` |
| 4 | [L04 RoPE](L04_rope/README.md) | cos and sin tables, rotation of q and k | `-L L04` |
| 5 ◆ | [L05 Attention](L05_attention/README.md) | causal grouped-query attention, two ways | `-L L05` |
| 6 | [L06 Block, model, loss](L06_model/README.md) | the 63.9 M model, loss, checkpoints | `-L L06` |
| 7 | [L07 Generation and KV cache](L07_generation/README.md) | sampling filters, the KV cache, generation | `-L L07` |

◆ long lesson: plan a weekend day. Week 2 (L08 training loop to L12 chat program, two
training runs, two reserve sessions) comes with milestone M1b.

## Before session 1

```bash
uv sync && cmake --preset clang && cmake --build --preset clang
ctest --preset clang -L LLM                     # all red: "not implemented: ..."
uv run tools/fetch_data.py tinystories-valid    # 22.5 MB of stories for L01 and L02
```

## Files of one lesson

| File | Writer | Content |
|------|--------|---------|
| `README.md`, `figure.svg` | course | the lesson page: picture, idea, what you write, check, sources |
| `<name>.py` | course | the Python reference: complete; run it with `uv run python -m LLM.<lesson>.<name>` |
| `<name>.hpp` | course | the C++ interface: names, shapes, behaviour, LibTorch hints |
| `<name>.cpp` | **you** | the bodies; each starts as `COURSE_TODO("name")` |
| `<name>_test.cpp`, `golden.py` | course | the test, and the golden file that the reference writes for it |

The line numbers in the "Sources" of each page refer to minimind commit `233d00fa06`
and LLMs-from-scratch commit `0311973e2c`.
