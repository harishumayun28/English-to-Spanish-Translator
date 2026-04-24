import streamlit as st
import torch
import pickle
import re
import sys
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="English → Spanish Translator",
    page_icon="🌎",
    layout="centered"
)

# ── Load transformer ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from rui_torch_transformer import Transformer, MultiHeadAttention

def _mha_forward(self, q, k, v):
    b, q_len, _ = q.shape
    kv_len = k.shape[1]
    q = self.W_q(q); k = self.W_k(k); v = self.W_v(v)
    q = q.view(b, q_len,  self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(b, kv_len, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(b, kv_len, self.n_heads, self.d_k).transpose(1, 2)
    scores = q @ k.transpose(2, 3)
    if self.is_causal:
        mask = self.mask.bool()[:q_len, :q_len]
        scores.masked_fill_(mask, -torch.inf)
    weights = torch.softmax(scores / self.d_k**0.5, dim=-1)
    weights = self.dropout(weights)
    out = (weights @ v).transpose(1, 2)
    out = out.contiguous().view(b, q_len, self.d_model)
    return self.out_proj(out)

MultiHeadAttention.forward = _mha_forward

# ── Tokenizer ─────────────────────────────────────────────────────────────────
def tokenize(text, lang='en'):
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿¡;:])", r" \1 ", text)
    allowed = r'[^a-z?.!,¿¡;: ]+' if lang == 'en' else r'[^a-záéíóúüñ?.!,¿¡;: ]+'
    text = re.sub(allowed, ' ', text)
    return text.split()

# ── Load pipeline (cached so it only loads once) ──────────────────────────────
@st.cache_resource
def load_pipeline():
    device = torch.device('cpu')
    save_dir = Path(__file__).parent / 'saved_model'

    with open(save_dir / 'src_vocab.pkl', 'rb') as f: sv  = pickle.load(f)
    with open(save_dir / 'tgt_vocab.pkl', 'rb') as f: tv  = pickle.load(f)
    with open(save_dir / 'config.pkl',    'rb') as f: cfg = pickle.load(f)

    mdl = Transformer(
        n_layers       = cfg['n_layers'],
        d_emb          = cfg['d_emb'],
        n_heads        = cfg['n_heads'],
        d_ff           = cfg['d_ff'],
        src_vocab_size = cfg['src_vocab_size'],
        tgt_vocab_size = cfg['tgt_vocab_size'],
        seq_len        = cfg['max_len'] + 2,
    ).to(device)

    mdl.load_state_dict(
        torch.load(save_dir / 'translator_weights.pt', map_location=device)
    )
    mdl.eval()
    return mdl, sv, tv, cfg, device

# ── Translate ─────────────────────────────────────────────────────────────────
def translate(sentence, mdl, sv, tv, cfg, device, max_new=80):
    tokens  = tokenize(sentence, 'en')
    src_ids = sv.encode(tokens, add_eos=True)
    if not src_ids:
        return ''
    max_src = cfg['max_len']
    if len(src_ids) > max_src:
        src_ids = src_ids[:max_src - 1] + [sv.EOS]
    src_t   = torch.tensor([src_ids], dtype=torch.long).to(device)
    tgt_ids = [tv.SOS]
    max_tgt = cfg['max_len']
    with torch.no_grad():
        for _ in range(max_new):
            tgt_in    = tgt_ids[-max_tgt:]
            tgt_t     = torch.tensor([tgt_in], dtype=torch.long).to(device)
            logits    = mdl((src_t, tgt_t))
            logit_vec = logits[0, -1, :].clone()
            for prev_tok in set(tgt_ids):
                logit_vec[prev_tok] *= 0.7
            nxt = logit_vec.argmax().item()
            tgt_ids.append(nxt)
            if nxt == tv.EOS:
                break
    return tv.decode(tgt_ids)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌎 English → Spanish Translator")
st.markdown(
    "Powered by a **Transformer** trained from scratch on 118,964 sentence pairs.  \n"
    "Built for CIS433 — Simon Business School, University of Rochester."
)
st.divider()

with st.spinner("Loading model..."):
    mdl, sv, tv, cfg, device = load_pipeline()

st.success("Model loaded and ready.", icon="✅")

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Translate")
user_input = st.text_area(
    "Enter an English sentence:",
    placeholder="e.g. Where is the library?",
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    translate_btn = st.button("Translate", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if translate_btn and user_input.strip():
    with st.spinner("Translating..."):
        result = translate(user_input.strip(), mdl, sv, tv, cfg, device)
    st.subheader("Spanish Translation")
    st.markdown(f"""
    <div style='background-color:#1e3a5f;padding:20px;border-radius:10px;
                border-left:4px solid #4a9eff;margin-top:8px'>
        <span style='font-size:1.3em;color:#ffffff;font-weight:500'>{result}</span>
    </div>
    """, unsafe_allow_html=True)

elif translate_btn and not user_input.strip():
    st.warning("Please enter a sentence to translate.")

st.divider()

# ── Example sentences ─────────────────────────────────────────────────────────
st.subheader("Try these examples")
examples = [
    "Hello, how are you?",
    "I love learning Spanish.",
    "Where is the bathroom?",
    "What time is it?",
    "I want to go to the library.",
    "She reads a book every day.",
    "We need to leave now.",
    "Thank you very much.",
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    with cols[i % 2]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            with st.spinner("Translating..."):
                result = translate(ex, mdl, sv, tv, cfg, device)
            st.markdown(f"""
            <div style='background-color:#1e3a5f;padding:16px;border-radius:10px;
                        border-left:4px solid #4a9eff;margin-top:4px;margin-bottom:12px'>
                <small style='color:#aaaaaa'>English: {ex}</small><br>
                <span style='font-size:1.1em;color:#ffffff;font-weight:500'>{result}</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Model info ────────────────────────────────────────────────────────────────
with st.expander("Model Details"):
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Architecture | Transformer (Encoder-Decoder) |
    | Embedding dim (`d_emb`) | {cfg['d_emb']} |
    | Layers (`n_layers`) | {cfg['n_layers']} |
    | Attention heads (`n_heads`) | {cfg['n_heads']} |
    | Feed-forward dim (`d_ff`) | {cfg['d_ff']} |
    | Source vocab size | {cfg['src_vocab_size']:,} |
    | Target vocab size | {cfg['tgt_vocab_size']:,} |
    | Training pairs | 118,964 |
    | BLEU Score | 39.74 |
    | Trained on | Google Colab T4 GPU |
    """)
