"""Draw the pictures of the lesson pages: <block>/<lesson>/figure.svg.

Run:  uv run tools/make_figures.py      (needs Graphviz: the program dot)

Some pictures run the Python reference (for example the merges of L01), so a picture
always agrees with the code.
"""

import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM.L01_tokenizer.tokenizer import FIRST_MERGE, merge, pretokenize, train_bpe  # noqa: E402

BLUE, YELLOW, GREEN, PURPLE, GREY = "#d6e6f5", "#fdf2d0", "#d5f0dc", "#e8daef", "#eceff1"
INK, MUTED, LINE, RED = "#1c2833", "#566573", "#34495e", "#c0392b"
SANS, MONO = "Helvetica, Arial, sans-serif", "DejaVu Sans Mono, Menlo, Consolas, monospace"


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.w, self.h = width, height
        self.items = [f'<rect width="{width}" height="{height}" fill="white"/>']
        self.text(width / 2, 30, title, size=19, anchor="middle")

    def rect(self, x, y, w, h, fill=GREY, stroke=LINE, rx=4, dash=False, width=1.0):
        d = ' stroke-dasharray="4,3"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')

    def text(self, x, y, s, size=13, anchor="start", color=INK, mono=False, bold=False, italic=False):
        font = MONO if mono else SANS
        extra = (' font-weight="bold"' if bold else "") + (' font-style="italic"' if italic else "")
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
                          f'text-anchor="{anchor}" fill="{color}"{extra}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color=LINE, arrow=False, dash=False, width=1.2):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        d = ' stroke-dasharray="4,3"' if dash else ""
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                          f'stroke-width="{width}"{m}{d}/>')

    def path(self, d, color=LINE, fill="none", arrow=False, width=1.2):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.items.append(f'<path d="{d}" stroke="{color}" fill="{fill}" stroke-width="{width}"{m}/>')

    def cell(self, x, y, s, w=None, h=26, fill=GREY, mono=True, size=12, color=INK):
        w = w or max(26, 8 * len(s) + 12)
        self.rect(x, y, w, h, fill=fill)
        self.text(x + w / 2, y + h / 2 + 4.5, s, size=size, anchor="middle", mono=mono, color=color)
        return w

    def save(self, path: Path) -> None:
        marker = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                  f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/>'
                  '</marker></defs>')
        body = "\n".join(self.items)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                        f'width="{self.w}" height="{self.h}">\n{marker}\n{body}\n</svg>\n')


def show(token: bytes) -> str:
    return token.decode("utf-8", errors="replace").replace(" ", "␣")


def l01_tokenizer(out: Path) -> None:
    s = Svg(940, 420, "L01 Byte-level BPE: one tokenizer, 6400 ids")
    x = 30
    for w, ids, name, fill in [(190, "0 – 255", "bytes", BLUE), (190, "256 – 271", "control: <|im_start|> …", YELLOW),
                               (190, "272 – 527", "action: <|act_k|> (block 3)", GREEN),
                               (330, "528 – 6399", "merges, in the order of training", PURPLE)]:
        s.rect(x, 52, w, 46, fill=fill)
        s.text(x + w / 2, 71, ids, size=14, anchor="middle", bold=True)
        s.text(x + w / 2, 89, name, size=12, anchor="middle")
        x += w + 4
    text = "the the then"
    s.text(30, 136, f'Training on "{text}": count the adjacent pairs, merge the most frequent pair, repeat.', size=14)
    merges = train_bpe([text], FIRST_MERGE + 4)
    vocab = {i: bytes([i]) for i in range(256)}
    seqs = [[*w] for w in pretokenize(text)]
    notes = ["(t,h) 3 and (h,e) 3: a tie; bytes 't' > 'h', so (t,h) wins",
             "(th,e) 3", "(␣,the) 2", "(␣the,n) 1"]
    y = 156
    for step in range(len(merges) + 1):
        s.text(30, y + 18, "bytes" if step == 0 else f"after {FIRST_MERGE + step - 1}", size=12, color=MUTED)
        x = 120
        for seq in seqs:
            for t in seq:
                x += s.cell(x, y, show(vocab[t]), fill=BLUE if t < 256 else PURPLE) + 2
            x += 16
        if step < len(merges):
            a, b = merges[step]
            new = FIRST_MERGE + step
            vocab[new] = vocab[a] + vocab[b]
            s.text(x + 6, y + 18, f"→ {new} = {show(vocab[new])}", size=13, mono=True, bold=True)
            s.text(x + 150, y + 18, notes[step], size=12, color=MUTED, mono="␣" in notes[step])
            seqs = [merge(q, (a, b), new) for q in seqs]
        y += 40
    s.text(30, y + 22, "Encode: cut out the special texts, cut the rest into pre-tokens (a space and a word, or one space byte),",
           size=13)
    s.text(30, y + 42, "then apply the merges: always the pair of the lowest rank first.  Decode: join the bytes of the ids.",
           size=13)
    s.save(out)


def l02_token_data(out: Path) -> None:
    s = Svg(940, 400, "L02 Token files, pre-training samples, batches")
    samples = [["71", "604", "33", "980", "12"], ["530", "88", "1203"], ["9", "641", "77", "3010"]]
    fills = [BLUE, YELLOW, GREEN]
    s.text(30, 72, "tokens.bin", size=13, bold=True)
    s.text(30, 88, "uint16", size=11, color=MUTED)
    x, starts = 130, []
    for ids, fill in zip(samples, fills):
        starts.append(x)
        for t in ids:
            x += s.cell(x, 58, t, w=48, fill=fill)
    starts.append(x)
    s.text(30, 152, "offsets.bin", size=13, bold=True)
    s.text(30, 168, "uint64", size=11, color=MUTED)
    offsets, total = [], 0
    for ids in samples:
        offsets.append(total)
        total += len(ids)
    offsets.append(total)
    for i, (o, sx) in enumerate(zip(offsets, starts)):
        cx = 130 + i * 56
        s.cell(cx, 138, str(o), w=48, fill=GREY)
        s.line(cx + 24, 138, sx + 2, 86, arrow=True, color=MUTED)
    s.text(130 + 4 * 56 + 10, 156, "start of each sample, then the total: sample i = tokens[offsets[i] : offsets[i+1]]",
           size=12, color=MUTED)
    s.text(30, 210, "pretrain_sample(sample 1, max_len = 8)", size=14, bold=True)
    rows = [("input_ids", ["<|im_start|>", "530", "88", "1203", "<|im_end|>", "pad", "pad", "pad"]),
            ("labels", ["<|im_start|>", "530", "88", "1203", "<|im_end|>", "-100", "-100", "-100"])]
    for r, (name, cells) in enumerate(rows):
        y = 226 + r * 36
        s.text(30, y + 18, name, size=13, mono=True)
        x = 130
        for c in cells:
            fill = GREY if c in ("pad", "-100") else YELLOW
            x += s.cell(x, y, c, w=96, fill=fill, size=11, color=RED if c == "-100" else INK) + 2
    s.text(130, 320, "-100: the loss does not count this position (padding).  pad = <|endoftext|>", size=12, color=MUTED)
    s.text(30, 360, "make_batch(data, indices, max_len): the samples at the indices, stacked -> input_ids, labels: [B, max_len]",
           size=13)
    s.save(out)


def l04_rope(out: Path) -> None:
    s = Svg(940, 380, "L04 RoPE: each pair of a head vector turns by an angle that grows with the position")
    names = [f"x{i}" for i in range(8)]
    for i, n in enumerate(names):
        x = 40 + i * 50 + (16 if i >= 4 else 0)
        s.cell(x, 80, n, w=46, fill=BLUE if i < 4 else YELLOW)
    s.text(40 + 98, 70, "first half x1", size=12, anchor="middle", color=MUTED)
    s.text(40 + 316, 70, "second half x2", size=12, anchor="middle", color=MUTED)
    for i in range(4):
        a, b = 40 + i * 50 + 23, 40 + (i + 4) * 50 + 16 + 23
        s.path(f"M{a} 108 C{a} {128 + i * 12} {b} {128 + i * 12} {b} 108", color=MUTED)
    s.text(40, 172, "pair i = (x_i, x_i+D/2): head_dim / 2 pairs (here head_dim = 8: 4 pairs)", size=13, color=MUTED)
    s.text(40, 210, "θ_i = base^(-2i / head_dim);  angle of pair i at position p = p · θ_i", size=13)
    s.text(40, 232, "pair 0 turns fastest, the last pair turns slowest", size=13, color=MUTED)
    s.text(40, 262, "rotate_half([x1 | x2]) = [-x2 | x1]", size=13, mono=True)
    s.text(40, 286, "rope(x) = x · cos + rotate_half(x) · sin", size=13, mono=True)
    s.text(40, 310, "cos, sin: tables [max_len, head_dim], made one time (rope_tables)", size=13, mono=True)
    cx, cy, r = 720, 200, 115
    s.line(cx - r - 20, cy, cx + r + 20, cy, color=GREY)
    s.line(cx, cy + r + 20, cx, cy - r - 20, color=GREY)
    s.path(f"M{cx + r} {cy} A{r} {r} 0 1 0 {cx - r} {cy} A{r} {r} 0 1 0 {cx + r} {cy}", color="#bfc9ca")
    import math
    s.text(cx, 62, "one pair as a point in a plane", size=12, anchor="middle", color=MUTED)
    for ang, color, label in [(20, MUTED, "position 0"), (75, LINE, "position p")]:
        ex, ey = cx + r * math.cos(math.radians(ang)), cy - r * math.sin(math.radians(ang))
        s.line(cx, cy, ex, ey, color=color, arrow=True, width=1.8)
        s.text(ex + (8 if ang < 50 else -6), ey - 6, label, size=12, anchor="start" if ang < 50 else "end", color=color)
    a1, a2 = math.radians(20), math.radians(75)
    s.path(f"M{cx + 45 * math.cos(a1):.1f} {cy - 45 * math.sin(a1):.1f} A45 45 0 0 0 "
           f"{cx + 45 * math.cos(a2):.1f} {cy - 45 * math.sin(a2):.1f}", color=RED, width=1.6)
    s.text(cx + 52, cy - 50, "p · θ_i", size=13, color=RED, bold=True)
    s.text(cx, cy + r + 42, "q at position m, k at position n: q·k depends only on m − n", size=12, anchor="middle")
    s.save(out)


DOT_HEAD = """graph [bgcolor=white, pad=0.3, fontname=Helvetica, labelloc=t, fontsize=19, nodesep=0.3, ranksep=0.35];
node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=12, color="#34495e", fillcolor="#eef3f8", margin="0.12,0.06"];
edge [color="#34495e", fontname=Helvetica, fontsize=10];"""

L03_DOT = f"""digraph L03 {{ {DOT_HEAD}
graph [rankdir=LR, label="L03 RMSNorm and SwiGLU"];
subgraph cluster_ffn {{ label="SwiGLU feed-forward:  down(silu(gate(x)) ⊙ up(x)), no bias"; fontsize=13; style=rounded; color="#aab7b8";
  fx [label="x  [..., 768]", shape=plaintext, style=""];
  gate [label="gate_proj\\n768 → 2432", fillcolor="{BLUE}"]; up [label="up_proj\\n768 → 2432", fillcolor="{BLUE}"];
  silu [label="SiLU\\nz · sigmoid(z)"]; mul [label="⊙", shape=circle, fixedsize=true, width=0.4];
  down [label="down_proj\\n2432 → 768", fillcolor="{BLUE}"]; fy [label="y  [..., 768]", shape=plaintext, style=""];
  fx -> gate -> silu -> mul; fx -> up -> mul; mul -> down -> fy; }}
subgraph cluster_rms {{ label="RMSNorm:  y = weight · x / sqrt(mean(x²) + ε)"; fontsize=13; style=rounded; color="#aab7b8";
  rx [label="x  [..., dim]", shape=plaintext, style=""];
  sq [label="x²"]; mean [label="mean over\\nthe last dim"]; eps [label="+ ε"]; rs [label="rsqrt"];
  m1 [label="×", shape=circle, fixedsize=true, width=0.4]; m2 [label="× weight [dim]\\n(learned)", fillcolor="{BLUE}"];
  ry [label="y  [..., dim]", shape=plaintext, style=""];
  rx -> sq -> mean -> eps -> rs -> m1 -> m2 -> ry; rx -> m1 [style=dashed]; }}
}}"""

L05_DOT = f"""digraph L05 {{ {DOT_HEAD}
graph [rankdir=LR, label="L05 Causal grouped-query attention: 8 query heads share 4 key-value heads"];
x [label="x\\n[B, T, 768]", shape=plaintext, style=""];
q [label="q_proj\\n→ [B, 8, T, 96]", fillcolor="{BLUE}"]; k [label="k_proj\\n→ [B, 4, T, 96]", fillcolor="{BLUE}"];
v [label="v_proj\\n→ [B, 4, T, 96]", fillcolor="{BLUE}"];
qn [label="q_norm\\n(RMSNorm, L03)"]; kn [label="k_norm\\n(RMSNorm, L03)"];
qr [label="RoPE (L04)"]; kr [label="RoPE (L04)"];
cache [label="KV cache (L07)\\nnew k, v after the old ones\\n→ [B, 4, S, 96]", fillcolor="{PURPLE}", style="rounded,filled,dashed"];
att [label="repeat each kv head 2 times\\nscores = q k^T / sqrt(96)   [B, 8, T, S]\\n-∞ where the mask is false\\nsoftmax, then × v", fillcolor="{GREEN}"];
mask [shape=plaintext, style="", label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
  <tr><td colspan="6"><font point-size="11">mask: T = 3 queries, S = 5 keys</font></td></tr>
  <tr><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td>0</td><td>0</td></tr>
  <tr><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td>0</td></tr>
  <tr><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td><td bgcolor="#d5f0dc">1</td></tr>
  <tr><td colspan="6"><font point-size="11">query t sees keys 0 .. S−T+t</font></td></tr></table>>];
o [label="[B, T, 8 · 96]\\no_proj → [B, T, 768]", fillcolor="{BLUE}"];
x -> q -> qn -> qr -> att; x -> k -> kn -> kr -> cache -> att; x -> v -> cache; mask -> att [style=dashed]; att -> o;
}}"""

L06_DOT = f"""digraph L06 {{ {DOT_HEAD}
graph [rankdir=TB, label="L06 The model: 63.9 M parameters; the head uses the embedding weight", ranksep=0.28];
ids [label="ids\\n[B, T]", shape=plaintext, style=""];
emb [label="embed_tokens\\n6400 × 768", fillcolor="{BLUE}"];
subgraph cluster_block {{ label="Block × 8"; fontsize=13; style=rounded; color="#aab7b8";
  n1 [label="input_layernorm\\nRMSNorm"]; at [label="self_attn\\n(L05)", fillcolor="{GREEN}"];
  p1 [label="+", shape=circle, fixedsize=true, width=0.4];
  n2 [label="post_attention_layernorm\\nRMSNorm"]; mlp [label="mlp\\nSwiGLU (L03)", fillcolor="{GREEN}"];
  p2 [label="+", shape=circle, fixedsize=true, width=0.4];
  n1 -> at -> p1 -> n2 -> mlp -> p2; }}
nf [label="norm\\nRMSNorm"]; head [label="linear(x, embed weight)\\ntied head", fillcolor="{BLUE}"];
logits [label="logits\\n[B, T, 6400]", shape=plaintext, style=""];
loss [label="lm_loss: position t\\nagainst label t+1;\\n-100 does not count", fillcolor="{YELLOW}"];
ids -> emb -> n1; emb -> p1 [label="  x (residual)", style=dashed]; p1 -> p2 [label="  x (residual)", style=dashed];
p2 -> nf -> head -> logits -> loss; emb -> head [style=dashed, color="#c0392b", label="same weight", fontcolor="#c0392b", constraint=false];
}}"""

L07_DOT = f"""digraph L07 {{ {DOT_HEAD}
graph [rankdir=LR, label="L07 Generation: read the prompt one time, then one id at each step"];
subgraph cluster_pick {{ label="Next id from the logits of the last position"; fontsize=13; style=rounded; color="#aab7b8";
  l [label="logits\\n[B, V]", shape=plaintext, style=""]; t [label="÷ temperature"]; k [label="top-k:\\nkeep the k largest"];
  tp [label="top-p: keep the smallest\\ntop set with mass ≥ p"]; sm [label="softmax"];
  s [label="sample\\n(argmax when\\ntemperature = 0)", fillcolor="{YELLOW}"];
  l -> t -> k -> tp -> sm -> s; }}
subgraph cluster_steps {{ label="KV cache: each layer keeps k and v of all earlier positions"; fontsize=13; style=rounded; color="#aab7b8";
  p [label="prompt\\n[B, T]", shape=plaintext, style=""];
  f0 [label="Model.forward(prompt, cache)\\npositions 0 .. T-1", fillcolor="{GREEN}"];
  c0 [label="cache: S = T", fillcolor="{PURPLE}"];
  f1 [label="Model.forward(id, cache)\\nposition T", fillcolor="{GREEN}"];
  c1 [label="cache: S = T + 1", fillcolor="{PURPLE}"];
  f2 [label="Model.forward(id, cache)\\nposition T + 1", fillcolor="{GREEN}"]; more [label="...", shape=plaintext, style=""];
  p -> f0 -> c0 -> f1 -> c1 -> f2 -> more; }}
}}"""


def dot(source: str, out: Path) -> None:
    subprocess.run(["dot", "-Tsvg", "-o", str(out)], input=source.encode(), check=True)


def main() -> None:
    lessons = ROOT / "LLM"
    l01_tokenizer(lessons / "L01_tokenizer/figure.svg")
    l02_token_data(lessons / "L02_token_data/figure.svg")
    dot(L03_DOT, lessons / "L03_rmsnorm_swiglu/figure.svg")
    l04_rope(lessons / "L04_rope/figure.svg")
    dot(L05_DOT, lessons / "L05_attention/figure.svg")
    dot(L06_DOT, lessons / "L06_model/figure.svg")
    dot(L07_DOT, lessons / "L07_generation/figure.svg")
    print("wrote 7 figures")


if __name__ == "__main__":
    main()
