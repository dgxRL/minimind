"""Generate docs/architecture/*.html — hand-laid inline-SVG diagrams of the MiniMind model (dense / MoE / KDA hybrid / AttnRes).

Usage: python docs/architecture/build.py [outdir]   (default: this folder). Edit the figures here, not the HTML.
"""
import html, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
esc = html.escape


class Fig:
    def __init__(self, fid, w, h, label):
        self.id, self.w, self.h, self.label, self.p = fid, w, h, label, []

    def box(self, x, y, w, h, title, sub=None, cls='n', rx=8, tcls='t'):
        self.p.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>')
        subs = [sub] if isinstance(sub, str) else (sub or [])
        cx, n = x + w / 2, len(subs)
        ty = y + h / 2 + 4.5 - n * 7.5
        self.text(cx, ty, title, tcls)
        for i, s in enumerate(subs):
            self.text(cx, ty + 16 + i * 15, s, 's')

    def text(self, x, y, s, cls='t', anchor='middle'):
        self.p.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(s)}</text>')

    def arrow(self, pts, label=None, at=None, cls='e', anchor='middle', start=False, head=True):
        d = ' '.join(f'{x},{y}' for x, y in pts)
        m = f' marker-end="url(#ar-{self.id})"' if head else ''
        if start: m += f' marker-start="url(#as-{self.id})"'
        self.p.append(f'<polyline class="{cls}" points="{d}"{m}/>')
        if label: self.text(at[0], at[1], label, 'el', anchor)

    def circle(self, cx, cy, r, s, cls='n'):
        self.p.append(f'<circle class="{cls}" cx="{cx}" cy="{cy}" r="{r}"/>')
        self.text(cx, cy + 5, s, 'tb')

    def grid(self, x, y, shades, cols=6, cell=16, uniform=None):
        for i, a in enumerate(shades):
            for j in range(cols):
                self.p.append(f'<rect class="cell" x="{x + j * cell}" y="{y + i * cell}" width="{cell}" height="{cell}" fill-opacity="{a:.2f}"/>')

    def raw(self, s):
        self.p.append(s)

    def render(self, caption):
        defs = (f'<defs><marker id="ar-{self.id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>'
                f'<marker id="as-{self.id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>')
        return (f'<figure><div class="scroll"><svg class="d" viewBox="0 0 {self.w} {self.h}" style="min-width:{int(self.w * 0.62)}px" role="img" aria-label="{esc(self.label)}">'
                f'{defs}{"".join(self.p)}</svg></div><figcaption>{caption}</figcaption></figure>')


CSS = """
:root{--bg:#ffffff;--fg:#1a1d23;--muted:#5b6472;--border:#d3d8e0;--surface:#f6f8fa;--node:#ffffff;
--attn:#2563eb;--attn-soft:#dbeafe;--kda:#c2410c;--kda-soft:#ffedd5;--moe:#047857;--moe-soft:#d1fae5;
--norm:#6b7280;--norm-soft:#eef0f3;--code-bg:#f3f4f6;--res:#7c3aed;--res-soft:#ede9fe}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0f1115;--fg:#e6e8eb;--muted:#9aa3af;--border:#343a45;--surface:#161a21;--node:#1b1f27;
--attn:#60a5fa;--attn-soft:#1b2a45;--kda:#fb923c;--kda-soft:#3b2516;--moe:#34d399;--moe-soft:#11332a;--norm:#9aa3af;--norm-soft:#232831;--code-bg:#1b1f27;--res:#a78bfa;--res-soft:#2a2140}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding-block:28px 80px;padding-inline:20px}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.88rem;border-bottom:1px solid var(--border);padding-bottom:10px;margin-bottom:8px}
nav a{color:var(--muted);text-decoration:none}nav a.on{color:var(--fg);font-weight:600}nav a:hover{color:var(--attn)}
h1{font-size:1.7rem;margin:.6rem 0 .2rem}h2{font-size:1.25rem;margin-top:2.4rem}
.lead{color:var(--muted);margin-top:0}
code{font:.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code-bg);border-radius:4px;padding:1px 5px}
pre{background:var(--code-bg);border-radius:8px;padding:12px 14px;overflow-x:auto;font-size:.85rem;line-height:1.5}pre code{padding:0;background:none}
figure{margin:1.2rem 0 1.6rem;border:1px solid var(--border);border-radius:10px;background:var(--surface);padding:14px}
figcaption{font-size:.86rem;color:var(--muted);margin-top:10px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.8rem 0}
.tw{overflow-x:auto}
th,td{border:1px solid var(--border);padding:6px 10px;text-align:left;vertical-align:top}th{background:var(--surface)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.key{display:inline-block;width:.8em;height:.8em;border-radius:2px;vertical-align:-1px;margin-right:4px;border:1.5px solid}
.k-attn{background:var(--attn-soft);border-color:var(--attn)}.k-kda{background:var(--kda-soft);border-color:var(--kda)}.k-moe{background:var(--moe-soft);border-color:var(--moe)}.k-res{background:var(--res-soft);border-color:var(--res)}
svg.d{display:block;width:100%;height:auto;color:var(--fg)}
svg.d text{font-family:inherit}
.n{fill:var(--node);stroke:var(--border);stroke-width:1.5}
.na{fill:var(--attn-soft);stroke:var(--attn);stroke-width:1.5}
.nk{fill:var(--kda-soft);stroke:var(--kda);stroke-width:1.5}
.nm{fill:var(--moe-soft);stroke:var(--moe);stroke-width:1.5}
.nr{fill:var(--res-soft);stroke:var(--res);stroke-width:1.5}
.rc{fill:var(--res);stroke:var(--border);stroke-width:1}
.nn{fill:var(--norm-soft);stroke:var(--norm);stroke-width:1.2}
.nd{fill:var(--norm-soft);stroke:var(--norm);stroke-width:1.2;stroke-dasharray:5 4}
.ng{fill:none;stroke:var(--norm);stroke-width:1.2;stroke-dasharray:6 4}
.cell{fill:var(--kda);stroke:var(--border);stroke-width:1}
.t{font-size:13px;fill:var(--fg)}.tb{font-size:13px;font-weight:600;fill:var(--fg)}
.s{font-size:11px;fill:var(--muted)}.el{font-size:11px;fill:var(--muted)}
.e{stroke:currentColor;stroke-width:1.4;fill:none}
.ed{stroke:currentColor;stroke-width:1.3;fill:none;stroke-dasharray:5 4}
"""

PAGES = [('index.html', 'Overview'), ('block_attention.html', 'Block & Attention'), ('kda.html', 'Kimi Delta Attention'),
         ('ffn_moe.html', 'FFN & MoE'), ('attnres.html', 'Attention Residuals'), ('inference.html', 'Cache & Generation')]


def page(fname, title, lead, body):
    nav = ''.join(f'<a href="{f}"{" class=on" if f == fname else ""}>{esc(t)}</a>' for f, t in PAGES)
    doc = (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>MiniMind {esc(title)}</title><style>{CSS}</style></head><body><div class="wrap">'
           f'<nav>{nav}<a href="../model_architecture.html">Original doc (pre-KDA)</a></nav>'
           f'<h1>{esc(title)}</h1><p class="lead">{lead}</p>{body}</div></body></html>\n')
    with open(os.path.join(OUT, fname), 'w') as f: f.write(doc)


# ───────────────────────────── index.html ─────────────────────────────
f = Fig('var', 950, 440, 'Four model variants as 8-layer stacks: use_kda swaps the token mixer of layers 0-2 and 4-6 to KDA, use_moe swaps every FFN to MoE')
variants = [('Dense', 'use_moe=0 · use_kda=0', False, False, '63.9M params'),
            ('MoE', 'use_moe=1 · use_kda=0', True, False, '198.4M total · 63.9M active'),
            ('KDA hybrid', 'use_moe=0 · use_kda=1', False, True, '69.3M params'),
            ('MoE + KDA', 'use_moe=1 · use_kda=1', True, True, '203.8M total · 69.3M active')]
for l in range(8): f.text(36, 98 + l * 34 + 18.5, f'L{l}', 's')
for i, (name, flags, moe, kda, params) in enumerate(variants):
    x = 60 + i * 222
    f.box(x, 10, 200, 44, name, flags, 'n', tcls='tb')
    f.box(x, 64, 200, 24, 'embed_tokens', None, 'nn', rx=5, tcls='s')
    for l in range(8):
        y = 98 + l * 34
        is_kda = kda and (l + 1) % 4 != 0
        if is_kda: f.box(x, y, 96, 28, 'KDA', None, 'nk', rx=5)
        else: f.box(x, y, 96, 28, 'Attn · NoPE' if kda else 'Attn · RoPE', None, 'na', rx=5)
        f.box(x + 104, y, 96, 28, 'MoE · top-1' if moe else 'SwiGLU', None, 'nm' if moe else 'n', rx=5)
    f.box(x, 372, 200, 24, 'norm → lm_head (tied)', None, 'nn', rx=5, tcls='s')
    f.text(x + 100, 420, params, 't')
fig_variants = f.render('The flags are independent; <code>use_attnres</code> (not drawn here) can be added to any of the four, see <a href="attnres.html">Attention Residuals</a>. <code>use_kda</code> replaces the token mixer with Kimi Delta Attention in every layer except each 4th one '
                        '(<code>full_attn_interval=4</code> → 3 KDA : 1 attention, the ratio Kimi Linear found best), and those remaining attention layers drop RoPE. '
                        '<code>use_moe</code> replaces every FFN with a 4-expert top-1 MoE.')

f = Fig('fwd', 960, 270, 'Forward pass: input ids, tied embedding, 8 blocks reading RoPE slices and the cache, final norm, lm_head, logits and loss')
f.box(10, 110, 100, 56, 'input_ids', '(B, T)')
f.box(140, 110, 130, 56, 'embed_tokens', '6400 × 768')
f.box(300, 110, 200, 56, '8 × MiniMindBlock', ['mixer = layer_types[i]', 'MoE layers emit aux_loss'], 'n', tcls='tb')
f.box(530, 110, 90, 56, 'norm', 'RMSNorm', 'nn')
f.box(650, 110, 120, 56, 'lm_head', '768 → 6400')
f.box(800, 110, 150, 56, 'logits', '(B, T, 6400)')
f.box(300, 14, 200, 50, 'freqs_cos / freqs_sin', '[start_pos : start_pos + T]', 'nn')
f.box(560, 14, 210, 50, 'past_key_values', 'one entry per layer', 'nn')
f.box(790, 206, 160, 50, 'loss', 'CE(shift) + Σ aux_loss')
f.arrow([(110, 138), (140, 138)]); f.arrow([(270, 138), (300, 138)]); f.arrow([(500, 138), (530, 138)])
f.arrow([(620, 138), (650, 138)]); f.arrow([(770, 138), (800, 138)]); f.arrow([(875, 166), (875, 206)], 'labels', (884, 190), anchor='start')
f.arrow([(400, 64), (400, 110)], 'RoPE slice (unused when NoPE)', (392, 92), anchor='end')
f.arrow([(665, 64), (665, 88), (470, 88), (470, 110)], start=True)
f.text(668, 82, 'read & write', 'el', 'start')
f.arrow([(710, 166), (710, 196), (205, 196), (205, 166)], cls='ed', head=False)
f.text(457, 212, 'same weight matrix (tie_word_embeddings)', 'el')
fig_fwd = f.render('Top-level data flow of <code>MiniMindForCausalLM</code>. <code>start_pos</code> comes from <code>get_past_length(past_key_values)</code>, '
                   'which works for both cache types (see <a href="inference.html">Cache &amp; Generation</a>). With <code>use_attnres</code> the blocks pass a list of '
                   'block representations instead of one hidden state, and a final AttnRes op mixes them before <code>norm</code> '
                   '(see <a href="attnres.html">Attention Residuals</a>).')

page('index.html', 'Model Architecture', 'Diagrams of <code>model/model_minimind.py</code> after adding Kimi Delta Attention '
     '(<a href="https://arxiv.org/abs/2510.26692">Kimi Linear, arXiv:2510.26692</a>) and Attention Residuals '
     '(<a href="https://arxiv.org/abs/2603.15031">AttnRes, arXiv:2603.15031</a>). Each page draws one part; all numbers are for the default config '
     '(hidden 768, 8 layers, vocab 6400). For how the training and inference code is organised, see '
     '<a href="../code_structure.md">code_structure.md</a>.',
     f"""<h2>Variants</h2>{fig_variants}
<p><span class="key k-attn"></span>full attention &nbsp; <span class="key k-kda"></span>KDA &nbsp; <span class="key k-moe"></span>MoE</p>
<h2>Forward pass</h2>{fig_fwd}
<h2>Config knobs added for KDA and AttnRes</h2>
<div class="tw"><table><tr><th>Field</th><th>Default</th><th>Meaning</th></tr>
<tr><td><code>use_kda</code></td><td>False</td><td>Turn on the hybrid; trainers expose it as <code>--use_kda 1</code> and save as <code>*_kda.pth</code></td></tr>
<tr><td><code>full_attn_interval</code></td><td>4</td><td>Every N-th layer keeps full attention; ≤ 0 means all layers are KDA</td></tr>
<tr><td><code>layer_types</code></td><td>derived</td><td>Explicit list of <code>"kda"</code> / <code>"full_attention"</code>; overrides the interval</td></tr>
<tr><td><code>kda_num_heads</code> / <code>kda_head_dim</code></td><td>8 / 96</td><td>KDA heads; state per head is head_dim × head_dim</td></tr>
<tr><td><code>kda_conv_size</code></td><td>4</td><td>Kernel width of the causal short convolution on q, k, v</td></tr>
<tr><td><code>attn_nope</code></td><td>= use_kda</td><td>Full-attention layers skip RoPE; KDA carries position information</td></tr>
<tr><td><code>use_attnres</code></td><td>False</td><td>Replace residual adds with Block Attention Residuals; <code>--use_attnres 1</code>, saved as <code>*_attnres.pth</code></td></tr>
<tr><td><code>attnres_block_size</code></td><td>2</td><td>Sub-layers per AttnRes block (attention and MLP count separately): 1 = Full AttnRes, 2 = one block per layer</td></tr></table></div>
<h2>Parameters per module</h2>
<div class="tw"><table><tr><th>Module</th><th class="num">Params</th><th>Notes</th></tr>
<tr><td>embed_tokens (tied with lm_head)</td><td class="num">4,915,200</td><td>6400 × 768</td></tr>
<tr><td>Attention</td><td class="num">1,769,664</td><td>q 768², k/v 768×384, o 768², q/k-norm</td></tr>
<tr><td>KimiDeltaAttention</td><td class="num">2,671,208</td><td>q/k/v/o 768², conv 2304×4, low-rank f/g 2×(768×96+96×768), b 768×8, A_log, dt_bias</td></tr>
<tr><td>FeedForward (SwiGLU)</td><td class="num">5,603,328</td><td>3 × 768 × 2432</td></tr>
<tr><td>MOEFeedForward</td><td class="num">22,416,384</td><td>4 experts + router 768 × 4; 1 expert active per token</td></tr>
<tr><td>AttnRes (× 17 with use_attnres)</td><td class="num">1,536</td><td>pseudo-query 768 + key RMSNorm 768; 2 per layer + 1 final = 26,112 total</td></tr></table></div>""")

# ───────────────────────────── block_attention.html ─────────────────────────────
f = Fig('blk', 720, 590, 'MiniMindBlock: pre-norm, token mixer chosen by layer_types, residual add, pre-norm, FFN or MoE, residual add')
f.box(200, 10, 200, 36, 'hidden_states h')
f.box(200, 76, 200, 36, 'input_layernorm', None, 'nn')
f.raw('<rect class="ng" x="140" y="140" width="340" height="120" rx="10"/>')
f.text(310, 158, 'self_attn — chosen by layer_types[layer_id]', 's')
f.box(152, 168, 152, 78, 'Attention', ['GQA · QK-norm', 'RoPE or NoPE'], 'na')
f.box(316, 168, 152, 78, 'KimiDeltaAttention', ['delta rule, per-channel', 'decay, fixed state'], 'nk')
f.box(10, 184, 100, 40, 'past_key_value', 'in → out', 'nn')
f.arrow([(110, 204), (140, 204)], start=True)
f.circle(300, 290, 12, '+')
f.box(200, 330, 200, 36, 'post_attention_layernorm', None, 'nn')
f.raw('<rect class="ng" x="140" y="394" width="340" height="100" rx="10"/>')
f.text(310, 412, 'mlp — chosen by use_moe', 's')
f.box(152, 422, 152, 62, 'MOEFeedForward', '4 experts · top-1', 'nm')
f.box(316, 422, 152, 62, 'FeedForward', 'SwiGLU 768→2432→768')
f.box(10, 433, 100, 40, 'aux_loss', 'summed by model', 'nn')
f.arrow([(152, 453), (110, 453)])
f.circle(300, 524, 12, '+')
f.arrow([(300, 46), (300, 76)]); f.arrow([(300, 112), (300, 140)]); f.arrow([(300, 260), (300, 278)])
f.arrow([(300, 302), (300, 330)]); f.arrow([(300, 366), (300, 394)]); f.arrow([(300, 494), (300, 512)])
f.arrow([(300, 536), (300, 566)]); f.text(312, 580, 'h out', 's', 'start')
f.arrow([(400, 28), (560, 28), (560, 290), (312, 290)], 'residual 1', (568, 160), anchor='start')
f.arrow([(300, 316), (600, 316), (600, 524), (312, 524)], 'residual 2', (608, 420), anchor='start')
fig_blk = f.render('Every block is pre-norm with two residual adds. Only the <code>self_attn</code> slot differs between layer types: '
                   '<code>build_token_mixer(config, layer_id)</code> picks the module and <code>build_feed_forward(config)</code> picks the FFN. '
                   'Both mixers share the same <code>forward(x, position_embeddings, past_key_value, use_cache, attention_mask)</code> signature, '
                   'so the block code does not depend on the choice. With <code>use_attnres</code> the two residual adds are replaced by depth attention '
                   '(<a href="attnres.html">Attention Residuals</a>).')

f = Fig('att', 800, 630, 'Attention: q, k, v projections, per-head RMSNorm on q and k, optional RoPE, KV cache concat, repeat_kv, causal attention, o_proj')
lanes = {'q': 170, 'k': 390, 'v': 610}
f.box(290, 10, 200, 36, 'x  (B, T, 768)')
for name, cx in lanes.items(): f.arrow([(390, 46), (390, 60), (cx, 60), (cx, 80)])
f.box(75, 80, 190, 46, 'q_proj', '768 → 8 heads × 96', 'na')
f.box(295, 80, 190, 46, 'k_proj', '768 → 4 heads × 96', 'na')
f.box(515, 80, 190, 46, 'v_proj', '768 → 4 heads × 96', 'na')
f.box(75, 152, 190, 40, 'q_norm', None, 'nn'); f.box(295, 152, 190, 40, 'k_norm', None, 'nn')
f.box(75, 218, 410, 40, 'RoPE  cos/sin[start_pos : +T]', 'skipped when attn_nope (default with use_kda)', 'nd')
f.box(295, 284, 410, 46, 'KV cache', 'k = cat(past_k, k) · v = cat(past_v, v)', 'nn')
f.arrow([(705, 307), (790, 307)]); f.text(790, 326, 'present (k, v)', 'el', 'end')
f.box(295, 356, 410, 40, 'repeat_kv × 2  →  8 heads', None, 'nn')
f.box(75, 424, 630, 84, 'causal attention', ['fast: F.scaled_dot_product_attention(is_causal=True) — when T > 1, no cache, mask all ones',
                                             'eager: softmax(QKᵀ/√96 + causal −∞ + (1 − mask)·−1e9) · V'], 'na')
f.box(290, 536, 200, 40, 'o_proj  768 → 768', None, 'na')
for a, b in [(126, 152), (192, 218), (258, 424)]: f.arrow([(170, a), (170, b)])
for a, b in [(126, 152), (192, 218), (258, 284), (330, 356), (396, 424)]: f.arrow([(390, a), (390, b)])
for a, b in [(126, 284), (330, 356), (396, 424)]: f.arrow([(610, a), (610, b)])
f.arrow([(390, 508), (390, 536)]); f.arrow([(390, 576), (390, 602)]); f.text(400, 620, 'output (B, T, 768)', 's', 'start')
fig_att = f.render('Full attention (GQA, 8 query heads sharing 4 KV heads). In the KDA hybrid only layers 3 and 7 use this module, and RoPE is bypassed: '
                   'the dashed box disappears from the path.')

page('block_attention.html', 'Block & Full Attention', 'How one <code>MiniMindBlock</code> is wired, and the full-attention token mixer it uses in dense layers.',
     f'<h2>MiniMindBlock</h2>{fig_blk}<h2>Attention</h2>{fig_att}')

# ───────────────────────────── kda.html ─────────────────────────────
f = Fig('kda', 1040, 696, 'KimiDeltaAttention dataflow: q k v projections with short conv and L2 norm, low-rank per-channel decay gate, beta, KDA core with recurrent state, head-wise RMSNorm with sigmoid output gate, o_proj')
L = {'q': 190, 'k': 320, 'v': 450, 'a': 610, 'b': 770, 'g': 930}
f.box(132, 10, 868, 34, 'x  (B, T, 768)')
for cx in L.values(): f.arrow([(cx, 44), (cx, 74)])
for n in 'qkv': f.box(L[n] - 58, 74, 116, 46, f'{n}_proj', '768 → 8 × 96')
f.box(L['a'] - 70, 74, 140, 46, 'f_proj (low rank)', '768 → 96 → 768', 'nk')
f.box(L['b'] - 70, 74, 140, 46, 'b_proj', '768 → 8 (per head)', 'nk')
f.box(L['g'] - 70, 74, 140, 46, 'g_proj (low rank)', '768 → 96 → 768, bias', 'nk')
f.box(132, 146, 376, 26, 'padding positions → 0', None, 'nn', rx=5, tcls='s')
f.box(132, 198, 376, 46, 'ShortConvolution + SiLU', 'depthwise causal conv, kernel 4, over [q | k | v] = 2304 ch', 'nk')
f.box(10, 198, 100, 46, 'conv_state', '(B, 3, 2304)', 'nn'); f.arrow([(110, 221), (132, 221)], start=True)
f.box(L['a'] - 70, 198, 140, 46, '+ dt_bias, softplus', 'per channel (768)', 'nk')
f.box(L['b'] - 70, 198, 140, 46, 'sigmoid', 'β ∈ (0, 1)', 'nk')
f.box(L['q'] - 58, 270, 116, 40, 'L2 norm', None, 'nn'); f.box(L['k'] - 58, 270, 116, 40, 'L2 norm', None, 'nn')
f.box(L['a'] - 70, 270, 140, 40, '× −exp(A_log)', 'g = log α ≤ 0', 'nk')
f.box(540, 336, 300, 26, 'padding positions → g = 0, β = 0', None, 'nn', rx=5, tcls='s')
f.box(132, 388, 708, 74, 'KDA core   (chunk_kda · fused_recurrent_kda · torch fallback)',
      ['S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ ;  o_t = S_tᵀ q_t / √96'], 'nk', tcls='tb')
f.box(10, 402, 100, 46, 'recurrent_state', 'S: (B, 8, 96, 96)', 'nn'); f.arrow([(110, 425), (132, 425)], start=True)
f.box(380, 488, 220, 40, 'o_norm', 'RMSNorm per head (96)', 'nn')
f.box(L['g'] - 70, 488, 140, 40, 'sigmoid', 'output gate', 'nk')
f.circle(490, 566, 13, '×')
f.box(390, 600, 200, 40, 'o_proj  768 → 768')
q, k, v, a, b, g = L.values()
for cx in (q, k):
    for y0, y1 in [(120, 146), (172, 198), (244, 270), (310, 388)]: f.arrow([(cx, y0), (cx, y1)])
for y0, y1 in [(120, 146), (172, 198), (244, 388)]: f.arrow([(v, y0), (v, y1)])
for y0, y1 in [(120, 198), (244, 270), (310, 336), (362, 388)]: f.arrow([(a, y0), (a, y1)])
for y0, y1 in [(120, 198), (244, 336), (362, 388)]: f.arrow([(b, y0), (b, y1)])
f.arrow([(g, 120), (g, 488)]); f.arrow([(g, 528), (g, 566), (503, 566)])
for n, lbl in zip('qkvab', ['q', 'k', 'v', 'g', 'β']): f.text(L[n] + 7, 380, lbl, 'el', 'start')
f.arrow([(490, 462), (490, 488)]); f.arrow([(490, 528), (490, 553)]); f.arrow([(490, 579), (490, 600)])
f.arrow([(490, 640), (490, 664)]); f.text(500, 684, 'output (B, T, 768)', 's', 'start')
fig_kda = f.render('One KDA layer, following Kimi Linear Fig. 3 / Eq. 10 and fla\'s <code>KimiDeltaAttention</code>. Orange marks parts that do not exist in full attention. '
                   'The cache is the two grey boxes on the left; neither grows with sequence length.')

f = Fig('rec', 960, 380, 'One KDA step on a head state matrix: per-row decay by alpha, delta-rule rank-1 correction, then read with q; below, GDN decays all rows equally while KDA decays each key channel separately')
sh0 = [.55, .7, .45, .8, .6, .5]
alpha = [.95, .35, .8, .15, .6, .9]
sh1 = [s * a for s, a in zip(sh0, alpha)]
sh2 = [min(.95, s + .25) if i in (1, 4) else s for i, s in enumerate(sh1)]
f.grid(40, 50, sh0); f.text(88, 38, 'S_{t−1}', 'tb'); f.text(88, 166, 'K × V = 96 × 96 per head', 's')
f.arrow([(150, 98), (250, 98)], '① decay', (200, 88)); f.text(200, 116, 'Diag(α_t)', 'el')
f.grid(260, 50, sh1); f.text(308, 38, 'Diag(α_t) S_{t−1}', 'tb'); f.text(308, 166, 'row i scaled by α_t,i', 's')
for i, al in enumerate(alpha): f.text(364, 50 + i * 16 + 12, f'α={al}', 'el', 'start')
f.arrow([(420, 98), (530, 98)], '② delta-rule write', (475, 88)); f.text(475, 116, 'v̂ = Sᵀ k_t', 'el')
f.grid(540, 50, sh2); f.text(588, 38, 'S_t', 'tb'); f.text(588, 166, 'S += β_t k_t (v_t − v̂)ᵀ', 's')
f.arrow([(650, 98), (760, 98)], '③ read', (705, 88)); f.text(705, 116, 'o_t = S_tᵀ q_t', 'el')
f.grid(770, 90, [.6], cols=6); f.text(818, 78, 'o_t', 'tb'); f.text(818, 128, 'V = 96', 's')
f.raw('<line class="ed" x1="20" y1="196" x2="940" y2="196"/>')
f.grid(150, 230, [.35] * 6); f.text(198, 220, 'Gated DeltaNet: Diag → α_t · I', 'tb')
f.text(270, 270, 'one scalar α per head:', 's', 'start'); f.text(270, 286, 'every key channel forgets', 's', 'start'); f.text(270, 302, 'at the same rate', 's', 'start')
f.grid(560, 230, [s * .7 for s in alpha]); f.text(608, 220, 'KDA: Diag(α_t), α_t ∈ [0,1]^96', 'tb')
f.text(680, 270, 'one α per key channel:', 's', 'start'); f.text(680, 286, 'fine-grained forgetting that also', 's', 'start'); f.text(680, 302, 'encodes position (so attention can be NoPE)', 's', 'start')
fig_rec = f.render('What happens to one head\'s state per token (Kimi Linear Eq. 1). Cell shade is memory strength. The delta rule first reads what the state already predicts '
                   'for <code>k_t</code> and writes only the error, scaled by β. The comparison underneath is the change KDA makes over GDN: '
                   'the decay becomes a diagonal matrix instead of a scalar.')

f = Fig('chk', 960, 300, 'Chunkwise form: the sequence is cut into chunks; inside a chunk everything is parallel matmuls, between chunks only the state S is passed')
for i in range(4):
    x = 30 + i * 230
    f.raw(f'<rect class="ng" x="{x}" y="40" width="210" height="150" rx="10"/>')
    f.text(x + 105, 30, f'chunk {i}', 's')
    for j in range(8): f.raw(f'<rect class="nk" x="{x + 14 + j * 23}" y="52" width="19" height="19" rx="3"/>')
    f.box(x + 10, 84, 190, 94, 'parallel within chunk', ['A_kk, A_qk from Γ ⊙ K, Γ ⊙ Q', 'W, U = (I + L)⁻¹ diag(β) [Γ⊙K, V]', 'o = (Γ⊙Q) S + A_qk (U − W S)'], 'n')
for i in range(5):
    xb = 20 + i * 230
    f.box(xb - 18, 228, 36, 26, 'S' + '₀₁₂₃₄'[i], None, 'nn', rx=5)
    if i < 4:
        f.arrow([(xb + 18, 241), (xb + 212, 241)])
        f.arrow([(xb + 50, 228), (xb + 50, 190)], cls='ed')
f.text(480, 282, 'between chunks:  S ← Diag(γ_C) S + (Γ ⊙ K)ᵀ (U − W S)', 'el')
f.text(940, 14, 'chunk size C = 64 (fla) · 16 (torch fallback)', 's', 'end')
fig_chk = f.render('How training stays fast (Kimi Linear §3.1, Eq. 6–9). A token-by-token loop would be T sequential steps; the chunk form does T/C sequential steps. '
                   'Everything inside a chunk is matmuls, using the WY representation and a UT transform (one triangular solve). '
                   'fla does this in Triton (<code>chunk_kda</code>); <code>torch_chunk_kda</code> is the same algorithm in PyTorch.')

f = Fig('dsp', 900, 290, 'Kernel dispatch: fla and CUDA picks chunk_kda for training or long prefill, fused_recurrent_kda for short no-grad calls; otherwise torch recurrent or torch chunk')
f.box(20, 120, 170, 50, 'KDA core call', 'q, k, v, g, β, S₀')
f.box(240, 120, 170, 50, 'fla installed and', 'x.is_cuda ?', 'nn', rx=24)
f.box(470, 50, 180, 50, 'grad enabled', 'or T > 64 ?', 'nn', rx=24)
f.box(470, 200, 180, 50, 'T ≤ 16 ?', None, 'nn', rx=24)
f.box(710, 20, 170, 46, 'chunk_kda', 'Triton · training / prefill', 'nk')
f.box(710, 84, 170, 46, 'fused_recurrent_kda', 'Triton · decoding', 'nk')
f.box(710, 170, 170, 46, 'torch_recurrent_kda', 'per-token loop')
f.box(710, 234, 170, 46, 'torch_chunk_kda', 'C = 16 · slow, memory-heavy')
f.arrow([(190, 145), (240, 145)])
f.arrow([(410, 135), (440, 135), (440, 75), (470, 75)], 'yes', (446, 104), anchor='start')
f.arrow([(410, 155), (440, 155), (440, 225), (470, 225)], 'no', (446, 194), anchor='start')
f.arrow([(650, 65), (680, 65), (680, 43), (710, 43)], 'yes', (684, 58), anchor='start')
f.arrow([(650, 85), (680, 85), (680, 107), (710, 107)], 'no', (684, 98), anchor='start')
f.arrow([(650, 215), (680, 215), (680, 193), (710, 193)], 'yes', (684, 208), anchor='start')
f.arrow([(650, 235), (680, 235), (680, 257), (710, 257)], 'no', (684, 248), anchor='start')
fig_dsp = f.render('All four paths compute the same thing (tested to ~1e-7 in fp32; fla\'s TF32 chunk kernel to ~2e-4) and use the same state layout <code>[B, H, K, V]</code>, '
                   'so a cache produced by one path can be continued by another. With fla, q/k L2-norm is fused into the kernel.')

f = Fig('pad', 900, 200, 'Left padding: pad tokens get alpha=1 and beta=0, so the state passes through them unchanged and the first real token sees the same state as an unpadded sequence')
toks = ['pad', 'pad', 'pad', 't1', 't2', 't3', 't4']
for i, t in enumerate(toks):
    x = 40 + i * 120
    pad = t == 'pad'
    f.box(x, 20, 100, 34, t, None, 'nn' if pad else 'n', tcls='s' if pad else 't')
    f.text(x + 50, 86, 'α = 1, β = 0' if pad else 'α_t, β_t', 'el')
    f.text(x + 50, 101, 'conv input 0' if pad else '', 'el')
    f.arrow([(x + 50, 54), (x + 50, 72)]); f.arrow([(x + 50, 108), (x + 50, 130)])
    f.box(x, 130, 100, 34, 'S₀' if pad else 'S' + '₀₁₂₃₄'[int(t[1])], None, 'nn' if pad else 'nk')
    if i: f.arrow([(x - 20, 147), (x, 147)])
f.text(450, 190, 'S₀ = 0 survives the pads untouched → identical to running t1…t4 alone', 'el')
fig_pad = f.render('How <code>attention_mask</code> is honoured without unpadding. GRPO / PPO / agent rollouts left-pad prompts, and KDA has no attention matrix to mask, so '
                   'padded positions are made into no-ops instead: zero conv input, no decay, no write. In the tests, greedy generation matched with and without padding.')

page('kda.html', 'Kimi Delta Attention', '<code>KimiDeltaAttention</code>: a gated delta-rule linear attention with per-channel decay. It replaces softmax attention in 6 of 8 layers '
     'when <code>use_kda=True</code>. Instead of a KV cache that grows with every token, it keeps one 96 × 96 state matrix per head.',
     f'<h2>Layer dataflow</h2>{fig_kda}<h2>The recurrence, one token</h2>{fig_rec}<h2>Chunkwise parallel form</h2>{fig_chk}'
     f'<h2>Kernel dispatch</h2>{fig_dsp}<h2>Padding</h2>{fig_pad}')

# ───────────────────────────── ffn_moe.html ─────────────────────────────
f = Fig('ffn', 790, 200, 'SwiGLU FFN: gate and up projections to 2432, SiLU on gate, elementwise product, down projection to 768')
f.box(10, 80, 90, 40, 'x  (768)')
f.box(150, 30, 170, 46, 'gate_proj', '768 → 2432'); f.box(150, 124, 170, 46, 'up_proj', '768 → 2432')
f.box(370, 30, 90, 46, 'SiLU', None, 'nn'); f.circle(520, 100, 14, '×')
f.box(570, 77, 170, 46, 'down_proj', '2432 → 768')
f.arrow([(100, 100), (125, 100), (125, 53), (150, 53)]); f.arrow([(125, 100), (125, 147), (150, 147)])
f.arrow([(320, 53), (370, 53)]); f.arrow([(460, 53), (520, 53), (520, 86)]); f.arrow([(320, 147), (520, 147), (520, 114)])
f.arrow([(534, 100), (570, 100)]); f.arrow([(740, 100), (780, 100)])
fig_ffn = f.render('<code>FeedForward</code>: <code>down(SiLU(gate(x)) * up(x))</code>, no biases, 5.6M params. Each MoE expert is exactly this module.')

f = Fig('moe', 960, 370, 'MoE routing: gate softmax over 4 experts, top-1 with normalization, each expert processes its tokens, weighted index_add combine; aux loss from scores and load')
for i in range(6): f.box(20, 60 + i * 44, 60, 32, f't{i + 1}', None, 'n', rx=5)
f.box(110, 150, 150, 60, 'gate  768 → 4', 'softmax over experts', 'nm')
f.box(300, 150, 150, 60, 'top-1', 'w = score / Σ score', 'nm')
for e in range(4):
    y = 20 + e * 86
    f.box(520, y, 170, 66, f'expert {e}', 'SwiGLU · 5.6M', 'nm')
    f.arrow([(450, 180), (485, 180), (485, y + 33), (520, y + 33)])
    f.arrow([(690, y + 33), (715, y + 33), (715, 180), (740, 180)])
f.text(490, 14, 'only tokens with idx = e', 'el', 'start')
f.box(740, 150, 180, 60, 'combine', 'y.index_add_(tok, w · out)')
f.box(110, 290, 340, 60, 'aux_loss = 4 · 5e-4 · Σₑ loadₑ · mean_scoreₑ', 'loadₑ = share of tokens routed to e', 'nn')
for i in range(6): f.arrow([(80, 76 + i * 44), (94, 76 + i * 44)], head=False)
f.arrow([(94, 76), (94, 296)], head=False); f.arrow([(94, 180), (110, 180)]); f.arrow([(260, 180), (300, 180)]); f.arrow([(920, 180), (950, 180)])
f.arrow([(185, 210), (185, 290)], 'scores', (192, 254), anchor='start'); f.arrow([(375, 210), (375, 290)], 'idx', (382, 254), anchor='start')
fig_moe = f.render('<code>MOEFeedForward</code> (<code>use_moe=1</code>). Each token runs one of 4 experts, so active params per layer stay at one FFN while total params are 4×. '
                   'Experts that receive no tokens in a step get a zero-valued gradient term, which keeps DDP from hanging on unused parameters. '
                   'The aux loss pushes the router towards balanced load. MoE is independent of KDA and changes only the FFN slot.')

page('ffn_moe.html', 'FFN & Mixture of Experts', 'The channel-mixing half of each block. It is unchanged by KDA.', f'<h2>SwiGLU FeedForward</h2>{fig_ffn}<h2>MOEFeedForward</h2>{fig_moe}')

# ───────────────────────────── inference.html ─────────────────────────────
f = Fig('cache', 960, 330, 'past_key_values in the KDA hybrid: KDA layers hold a fixed-size dict, attention layers hold k and v tensors that grow by one row per token')
for l in range(8):
    y = 30 + l * 36
    f.text(30, y + 18.5, f'L{l}', 's')
    if (l + 1) % 4:
        f.box(60, y, 300, 28, 'conv_state · recurrent_state S · seen_tokens', None, 'nk', rx=5, tcls='s')
    else:
        f.box(60, y, 640, 28, '(k, v): (B, T, 4, 96) each — one more row per generated token', None, 'na', rx=5, tcls='s')
        f.raw(f'<rect class="nd" x="700" y="{y}" width="200" height="28" rx="5"/>')
        f.arrow([(780, y + 14), (890, y + 14)]); f.text(770, y + 18.5, 'T grows', 'el', 'end')
f.box(420, 24, 330, 40, 'get_past_length(past)', 'entry 0: seen_tokens (dict) or k.shape[1] (tuple)', 'n')
f.arrow([(420, 44), (360, 44)])
f.text(30, 18, 'past_key_values — one entry per layer', 's', 'start')
fig_cache = f.render('Cache layout for the default hybrid. A KDA entry is a dict: <code>conv_state</code> (B, 3, 2304) bf16, <code>recurrent_state</code> (B, 8, 96, 96) fp32, '
                     'and <code>seen_tokens</code>. Its size does not depend on T. The old code read the cache length from <code>past[0][0].shape[1]</code>, which breaks when layer 0 is KDA. '
                     '<code>get_past_length</code> replaces it in both <code>forward</code> and <code>generate</code>.')

f = Fig('gen', 960, 290, 'generate: prefill runs the chunk kernel and SDPA over the prompt; each decode step runs the fused recurrent kernel and appends one KV row')
f.raw('<rect class="ng" x="20" y="30" width="400" height="230" rx="10"/>'); f.text(220, 22, 'prefill: forward(input_ids[:, :P], use_cache=True)', 's')
f.box(40, 70, 170, 60, 'KDA layers', 'chunk_kda (P > 64) → S', 'nk')
f.box(230, 70, 170, 60, 'attention layers', 'SDPA → (k, v) for P', 'na')
f.box(120, 180, 200, 50, 'sample next token', 'top-k · top-p · temperature')
f.arrow([(125, 130), (125, 155), (220, 155), (220, 180)]); f.arrow([(315, 130), (315, 155), (220, 155)], head=False)
f.raw('<rect class="ng" x="460" y="30" width="480" height="230" rx="10"/>'); f.text(700, 22, 'decode step: forward(input_ids[:, past_len:])  — 1 token', 's')
f.box(480, 70, 210, 60, 'KDA layers', 'fused_recurrent_kda: S ← step(S)', 'nk')
f.box(710, 70, 210, 60, 'attention layers', 'append 1 row to (k, v)', 'na')
f.box(580, 180, 240, 50, 'sample · append to input_ids', 'attention_mask gets +1 column')
f.arrow([(585, 130), (585, 155), (700, 155), (700, 180)]); f.arrow([(815, 130), (815, 155), (700, 155)], head=False)
f.arrow([(820, 205), (932, 205), (932, 100), (920, 100)], 'next step', (926, 196), anchor='end')
f.arrow([(420, 145), (460, 145)], 'cache', (440, 136))
fig_gen = f.render('<code>MiniMindForCausalLM.generate</code>; each step picks a token with <code>sample_next_token</code> '
                   '(repetition penalty → top-k → top-p → sample or argmax). Per-token cost in a KDA layer is constant. In an attention layer it grows with context, so the hybrid keeps only 2 of 8 layers on the growing path.')

page('inference.html', 'Cache & Generation', 'What the model carries between decoding steps, and why the hybrid uses less memory at long context.',
     f"""<h2>Cache layout</h2>{fig_cache}
<h2>Cache size per sequence</h2>
<p>Attention: 2 × 4 heads × 96 × 2 bytes = 1,536 B per token per layer. KDA: 8 × 96 × 96 × 4 B (fp32 state) + 3 × 2304 × 2 B (conv) = 308,736 B per layer, independent of T.</p>
<div class="tw"><table><tr><th class="num">Context T</th><th class="num">Dense (8 attention)</th><th class="num">KDA hybrid (2 attention + 6 KDA)</th><th class="num">Ratio</th></tr>
<tr><td class="num">512</td><td class="num">6.3 MB</td><td class="num">3.4 MB</td><td class="num">1.8×</td></tr>
<tr><td class="num">2,048</td><td class="num">25.2 MB</td><td class="num">8.1 MB</td><td class="num">3.1×</td></tr>
<tr><td class="num">8,192</td><td class="num">100.7 MB</td><td class="num">27.0 MB</td><td class="num">3.7×</td></tr>
<tr><td class="num">32,768</td><td class="num">402.7 MB</td><td class="num">102.5 MB</td><td class="num">3.9×</td></tr></table></div>
<p>The ratio approaches 4× (8 / 2 attention layers) as T grows. Below about 200 tokens per KDA layer, the fixed KDA state is larger than the KV cache it replaces.</p>
<h2>Generation loop</h2>{fig_gen}""")
# ───────────────────────────── attnres.html ─────────────────────────────
f = Fig('rcmp', 940, 470, 'Standard residual versus Block AttnRes for one MiniMindBlock: the standard block adds into one stream; the AttnRes block reads a softmax mix of stored block representations and adds its outputs only into the current partial block sum')
# left: standard residual
f.raw('<rect class="ng" x="10" y="30" width="330" height="430" rx="10"/>'); f.text(175, 22, 'standard residual (use_attnres=0)', 's')
f.box(95, 50, 160, 34, 'h  (one stream)')
f.box(95, 112, 160, 34, 'input_layernorm', None, 'nn'); f.box(95, 168, 160, 40, 'self_attn', 'Attention or KDA', 'na')
f.circle(175, 236, 12, '+')
f.box(95, 272, 160, 34, 'post_attention_layernorm', None, 'nn'); f.box(95, 328, 160, 40, 'mlp', 'SwiGLU or MoE', 'nm')
f.circle(175, 396, 12, '+')
for a, b in [(84, 112), (146, 168), (208, 224), (248, 272), (306, 328), (368, 384), (408, 440)]: f.arrow([(175, a), (175, b)])
f.arrow([(255, 67), (300, 67), (300, 236), (187, 236)], 'h + attn', (306, 150), anchor='start')
f.arrow([(175, 254), (310, 254), (310, 396), (187, 396)], 'h + mlp', (316, 330), anchor='start')
f.text(185, 452, 'h out = emb + Σ all outputs, weight 1', 's', 'start')
# right: block attnres
f.raw('<rect class="ng" x="370" y="30" width="560" height="430" rx="10"/>'); f.text(650, 22, 'Block AttnRes (use_attnres=1, attnres_block_size=2)', 's')
f.box(385, 50, 150, 400, 'blocks', None, 'n', tcls='tb')
for i, (t, sub) in enumerate([('b₀', 'embedding'), ('b₁', 'layer 0 sum'), ('⋮', None), ('bₙ₋₁', 'layer l−1 sum')]):
    f.box(397, 90 + i * 62, 126, 48, t, sub, 'nn' if t != '⋮' else 'n', rx=6)
f.box(397, 360, 126, 48, 'bₙ  (appended)', 'attn + mlp of layer l', 'nr', rx=6)
f.box(625, 50, 190, 34, 'partial = None', 'block starts empty' if False else None, 'nn')
f.box(625, 104, 190, 40, 'AttnRes  (w_attn)', 'softmax over blocks', 'nr')
f.box(625, 158, 190, 30, 'input_layernorm', None, 'nn'); f.box(625, 202, 190, 40, 'self_attn', 'Attention or KDA', 'na')
f.box(625, 256, 190, 40, 'AttnRes  (w_mlp)', 'blocks + [attn_out]', 'nr')
f.box(625, 310, 190, 30, 'post_attention_layernorm', None, 'nn'); f.box(625, 354, 190, 40, 'mlp', 'SwiGLU or MoE', 'nm')
f.circle(720, 424, 12, '+')
for a, b in [(84, 104), (144, 158), (188, 202), (340, 354), (394, 412)]: f.arrow([(720, a), (720, b)])
f.arrow([(720, 242), (720, 256)]); f.arrow([(720, 296), (720, 310)])
f.arrow([(523, 124), (625, 124)], 'read all', (574, 116))
f.arrow([(535, 276), (625, 276)], 'read all', (580, 268))
f.arrow([(815, 222), (880, 222), (880, 424), (732, 424)], 'attn_out', (886, 300), anchor='start')
f.arrow([(815, 222), (850, 222), (850, 276), (815, 276)], head=True)
f.arrow([(708, 424), (523, 384)], 'close block', (610, 392))
fig_rcmp = f.render('Left: the input to every sub-layer is the running sum of everything before it. Right: each sub-layer gets its own input, '
                    'a softmax-weighted mix of the stored block representations (plus the current partial sum), and its output is added only to the '
                    'current block. With the default <code>attnres_block_size=2</code> a block is one Transformer layer (attention + MLP), so 8 layers '
                    'give 8 blocks plus the embedding, the paper\'s N ≈ 8.')

f = Fig('rop', 940, 400, 'One AttnRes op: each source v_i gets a cached 1/rms scale, a logit from the folded query (w times the norm weight), softmax over sources, and a weighted sum')
src = [('v₀ = b₀', 'embedding'), ('v₁ = b₁', 'block'), ('⋮', None), ('partial', 'current block, if any')]
for i, (t, sub) in enumerate(src):
    y = 40 + i * 66
    f.box(10, y, 150, 48, t, sub, 'nn' if t != '⋮' else 'n', rx=6)
    if t == '⋮': continue
    f.box(200, y, 150, 48, 'rᵢ = 1 / rms(vᵢ)', 'once per block' if t != 'partial' else 'on use', 'n', rx=6)
    f.box(390, y, 170, 48, 'ℓᵢ = rᵢ · (q · vᵢ)', None, 'nr', rx=6)
    f.arrow([(160, y + 24), (200, y + 24)]); f.arrow([(350, y + 24), (390, y + 24)])
    f.arrow([(560, y + 24), (600, y + 24)])
f.box(390, 318, 170, 48, 'q = w ⊙ g', 'pseudo-query × norm weight', 'nr')
f.arrow([(475, 318), (475, 286)], 'shared by all ℓᵢ · w starts at 0', (485, 306), anchor='start')
f.box(600, 40, 110, 246, 'softmax', ['over sources', 'αᵢ (fp32)'], 'nr')
f.box(750, 120, 180, 76, 'h = Σᵢ αᵢ vᵢ', ['input of the next', 'sub-layer norm'], 'nr')
f.arrow([(710, 158), (750, 158)])
f.arrow([(85, 286), (85, 384), (840, 384), (840, 196)], 'values vᵢ (unnormalized)', (250, 378))
fig_rop = f.render('<code>AttnRes.forward</code>. The paper\'s φ(w, k) = exp(wᵀ RMSNorm(k)) needs a normalized copy of every source at every op. Because '
                   'wᵀ(g ⊙ v / rms(v)) = ((w ⊙ g)ᵀ v) / rms(v), the code folds the norm weight g into the query and keeps only the per-token scale '
                   '1/rms(v). A completed block\'s scale is computed once, when the block closes, and shared by all later ops. No stacked or '
                   'normalized copies are kept for backward, so peak memory stays close to the baseline. Pseudo-queries start at zero, so every op begins as a plain average of its sources.')

def visibility(S, L=8):
    blocks, partial, rows = ['b0'], False, []
    for j in range(2 * L):
        rows.append((f'L{j // 2} {"attn" if j % 2 == 0 else "mlp"}', list(blocks), partial))
        partial = True
        if (j + 1) % S == 0: blocks.append(f'b{len(blocks)}'); partial = False
    rows.append(('final', list(blocks), partial))
    return rows, len(blocks)

cell, top = 14, 64
f = Fig('rmat', 940, top + 17 * cell + 40, 'Which sources each AttnRes op reads, for block sizes 1 (full), 2 (default) and 4: a staircase over block representations, plus the partial sum of the current block')
x0 = 80
for S, title in [(1, 'S = 1 · Full AttnRes'), (2, 'S = 2 · default'), (4, 'S = 4')]:
    rows, nb = visibility(S)
    cols = nb + (1 if S > 1 else 0)
    f.text(x0 + cols * cell / 2, 20, title, 'tb')
    for c in range(nb): 
        if c % 2 == 0: f.text(x0 + c * cell + cell / 2, top - 8, f'b{c}', 's')
    if S > 1: f.text(x0 + nb * cell + cell / 2, top - 8, 'p', 's')
    for r, (lab, bl, part) in enumerate(rows):
        y = top + r * cell
        if S == 1: f.text(x0 - 8, y + 11, lab, 's', 'end')
        for c in range(cols):
            on = (c < nb and f'b{c}' in bl) or (c == nb and part)
            cls = 'rc' if on else 'n'
            op = ' fill-opacity="0.45"' if on and c == nb else ''
            f.raw(f'<rect class="{cls}" x="{x0 + c * cell}" y="{y}" width="{cell}" height="{cell}"{op}/>')
    x0 += cols * cell + 70
f.text(470, top + 17 * cell + 28, 'rows: the 17 AttnRes ops (2 per layer + final)  ·  columns: stored blocks b₀…, and p = partial sum of the current block (lighter)', 's')
fig_rmat = f.render('The depth mixing pattern for 8 layers (16 sub-layers). Each filled cell is a source that op can weight. S = 1 stores every '
                    'sub-layer output (17 sources at the end); S = 2 stores one sum per layer, so an MLP sub-layer sees the finished blocks plus its own '
                    'layer\'s attention output; S = 4 groups two layers. N = 1 (S = 16) would be the standard residual with the embedding kept apart.')

page('attnres.html', 'Attention Residuals', 'Block AttnRes (<a href="https://arxiv.org/abs/2603.15031">arXiv:2603.15031</a>) replaces the fixed '
     'residual sum with softmax attention over depth: each sub-layer chooses how much of the embedding and of each earlier block to read. '
     'Enabled with <code>use_attnres=True</code>; works with any combination of <code>use_moe</code> and <code>use_kda</code>.',
     f"""<h2>Residual stream, before and after</h2>{fig_rcmp}
<h2>One AttnRes op</h2>{fig_rop}
<h2>What each op can see</h2>{fig_rmat}
<h2>Cost</h2>
<p>Parameters: 1,536 per op (pseudo-query + key norm), 17 ops, 26,112 in total (+0.04%). The KV / KDA cache is unchanged: AttnRes mixes only along depth, within each token, so incremental decoding needs no extra state.</p>
<div class="tw"><table><tr><th>Training step (B=32, T=340, bf16, RTX 4090)</th><th class="num">Baseline</th><th class="num">AttnRes S=2</th><th class="num">AttnRes S=4</th></tr>
<tr><td>eager</td><td class="num">73 ms · 5.5 GiB</td><td class="num">112 ms · 6.7 GiB</td><td class="num">101 ms · 6.7 GiB</td></tr>
<tr><td><code>--use_compile 1</code></td><td class="num">43 ms · 3.5 GiB</td><td class="num">54 ms · 4.0 GiB</td><td class="num">51 ms · 3.9 GiB</td></tr></table></div>
<p>Eager mode runs each source's weighted add as its own elementwise kernel, which a model this small notices. <code>torch.compile</code> fuses them, so train AttnRes models with <code>--use_compile 1</code>.</p>
<h2>Where it lives in the code</h2>
<div class="tw"><table><tr><th>Piece</th><th>Code</th></tr>
<tr><td>Depth attention op</td><td><code>AttnRes</code> (<code>query</code>, <code>norm</code>, <code>source()</code>)</td></tr>
<tr><td>Per-layer wiring</td><td><code>MiniMindBlock.forward_attn_res</code>; <code>attn_res_close</code> marks block boundaries</td></tr>
<tr><td>State threading and final mix</td><td><code>MiniMindModel.forward</code>: <code>attn_res_state = (blocks, partial)</code>, <code>final_attn_res</code></td></tr>
<tr><td>Trainers and scripts</td><td><code>--use_attnres 1</code> (distillation: <code>--student_use_attnres</code> / <code>--teacher_use_attnres</code>); weights get the <code>_attnres</code> suffix; Qwen3 export is refused</td></tr></table></div>""")
print('ok')
