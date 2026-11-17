"""Draws roadmap.svg. Run: python3 roadmap.py"""
W, H = 1040, 350
X0, BW, GAP, Y = 30, 225, 12, 78          # left margin, block width, gap, block top
blocks = [
    ("Block 1", "LLM", "weeks 1-2", "#d6e6f5", ["tokenizer", "attention, RoPE, GQA", "training, chat, think"],
     "A chat model that thinks", [9, 11], 12),
    ("Block 2", "VLM", "weeks 3-4", "#fde3c8", ["vision encoder", "projector, image tokens", "two training stages"],
     "A model that answers about an image", [7, 8], 11),
    ("Block 3", "VLA", "weeks 5-6", "#d5f0dc", ["action tokens, decision text", "flow-matching action expert", "reason, then act"],
     "A model that reasons, then acts", [4, 7], 11),
    ("Block 4", "Reasoning", "weeks 7-8", "#e8daef", ["answer checker", "more compute at inference", "reward training, distillation"],
     "Methods that make it reason better", [], 12),
]
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-size="20" fill="#1c2833">The shortest path: 14 sessions of two hours for each block</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-size="13" fill="#566573">Read the Python reference. Write the C++. The test compares the two.</text>']
for i, (blk, name, weeks, col, topics, product, nights, lessons) in enumerate(blocks):
    x = X0 + i * (BW + GAP)
    o.append(f'<rect x="{x}" y="{Y}" width="{BW}" height="164" rx="12" fill="{col}" stroke="#34495e" stroke-width="1.2"/>')
    o.append(f'<text x="{x+14}" y="{Y+24}" font-size="12" fill="#566573">{blk} - {weeks}</text>')
    o.append(f'<text x="{x+14}" y="{Y+52}" font-size="24" font-weight="bold" fill="#1c2833">{name}</text>')
    for k, t in enumerate(topics):
        o.append(f'<text x="{x+14}" y="{Y+78+k*18}" font-size="12.5" fill="#1c2833">- {t}</text>')
    o.append(f'<text x="{x+14}" y="{Y+136}" font-size="10" fill="#566573">PRODUCT</text>')
    o.append(f'<text x="{x+14}" y="{Y+152}" font-size="12" font-style="italic" fill="#1c2833">{product}</text>')
    # 14 sessions: 12 lessons + 2 reserve
    cw = (BW - 13 * 3) / 14
    for d in range(14):
        cx = x + d * (cw + 3)
        fill = col if d < lessons else "white"
        dash = '' if d < lessons else ' stroke-dasharray="3,2"'
        o.append(f'<rect x="{cx:.1f}" y="{Y+176}" width="{cw:.1f}" height="14" rx="3" fill="{fill}" stroke="#34495e" stroke-width="0.9"{dash}/>')
        if (d + 1) in nights:
            o.append(f'<circle cx="{cx+cw/2:.1f}" cy="{Y+202}" r="4" fill="#6c3483"/>')
    if i < len(blocks) - 1:
        ax = x + BW + 1
        o.append(f'<path d="M{ax} {Y+75} l{GAP-2} 0" stroke="#34495e" stroke-width="1.6" marker-end="url(#a)"/>')
o.insert(2, '<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#34495e"/></marker></defs>')
ly = Y + 232
o.append(f'<rect x="{X0}" y="{ly}" width="15" height="14" rx="3" fill="#d6e6f5" stroke="#34495e" stroke-width="0.9"/><text x="{X0+22}" y="{ly+11.5}" font-size="12" fill="#1c2833">lesson, one session</text>')
o.append(f'<rect x="{X0+170}" y="{ly}" width="15" height="14" rx="3" fill="white" stroke="#34495e" stroke-width="0.9" stroke-dasharray="3,2"/><text x="{X0+192}" y="{ly+11.5}" font-size="12" fill="#1c2833">reserve session</text>')
o.append(f'<circle cx="{X0+322}" cy="{ly+7}" r="4" fill="#6c3483"/><text x="{X0+333}" y="{ly+11.5}" font-size="12" fill="#1c2833">a training run starts and runs at night</text>')
o.append(f'<rect x="{X0+590}" y="{ly-4}" width="{W-X0-590-30}" height="22" rx="6" fill="#f2f3f4" stroke="#aab7b8" stroke-dasharray="4,3"/><text x="{X0+600}" y="{ly+11.5}" font-size="12" fill="#566573">Block 5 - DiffusionVL: reserved, planned after block 3</text>')
o.append('</svg>')
open("roadmap.svg", "w").write("\n".join(o) + "\n")
