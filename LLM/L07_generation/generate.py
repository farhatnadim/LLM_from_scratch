"""L07 Generation and KV cache (docs/specification.md, S-5).

The first step reads the whole prompt and fills the cache. Each next step reads one
new token: attention takes the keys and values of the earlier positions from the cache.
"""

import torch

from LLM.L05_attention.attention import LayerCache
from LLM.L06_model.model import Config, Model, init_weights


def filter_probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """logits [B, V] -> probabilities [B, V]. top_k = 0 and top_p = 1 turn a filter off."""
    logits = logits / temperature
    if top_k > 0:                                                         # keep the k largest
        kth = logits.topk(top_k, dim=-1).values[:, -1:]                   # [B, 1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:                                                       # keep the smallest top set with mass >= top_p
        sorted_logits, order = logits.sort(dim=-1, descending=True)       # [B, V]
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p > top_p                               # mass before the token
        logits = logits.scatter(-1, order, sorted_logits.masked_fill(drop, float("-inf")))
    return logits.softmax(dim=-1)                                         # [B, V]


@torch.no_grad()
def generate(model: Model, ids: torch.Tensor, max_new: int, temperature: float = 0.0, top_k: int = 0,
             top_p: float = 1.0, eos: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """ids: prompt [B, T] -> new ids [B, n], n <= max_new. Greedy when temperature is 0.
    A row that wrote eos writes eos after it. The loop stops when all rows wrote eos."""
    cache = [LayerCache() for _ in model.layers]
    logits = model(ids, cache)[:, -1]                                     # [B, V]: the whole prompt at once
    done = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)  # [B]
    out = []
    for _ in range(max_new):
        if temperature == 0:
            nxt = logits.argmax(dim=-1)                                   # [B]
        else:
            probs = filter_probs(logits, temperature, top_k, top_p)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)  # [B]
        if eos is not None:
            nxt = nxt.masked_fill(done, eos)
            done |= nxt == eos
        out.append(nxt)
        if done.all():
            break
        logits = model(nxt[:, None], cache)[:, -1]                        # [B, V]: one new position
    return torch.stack(out, dim=1)                                        # [B, n]


if __name__ == "__main__":
    import time

    model = Model(Config()).eval()
    init_weights(model)
    prompt = torch.randint(0, 6400, (1, 32))
    start = time.perf_counter()
    fast = generate(model, prompt, 64)
    print(f"64 tokens with the KV cache: {time.perf_counter() - start:.2f} s")
    start, ids = time.perf_counter(), prompt
    with torch.no_grad():
        for _ in range(64):                    # without a cache: each step reads the whole sequence again
            ids = torch.cat([ids, model(ids)[:, -1:].argmax(dim=-1)], dim=1)
    print(f"64 tokens without a cache: {time.perf_counter() - start:.2f} s; same ids: {torch.equal(ids[:, 32:], fast)}")
