# English → Spanish Transformer Translator

A neural machine translator built from scratch using a Transformer architecture, trained on 118,964 English-Spanish sentence pairs.

## Live Demo
👉 [Open on Streamlit Cloud](https://your-app-url.streamlit.app)

## Model
- Architecture: Transformer Encoder-Decoder (`rui_torch_transformer.py`)
- Config: `d_emb=128`, `n_layers=2`, `n_heads=8`, `d_ff=512`
- Training data: Tatoeba `spa.txt` — 118,964 pairs
- BLEU Score: **39.74**
- Trained on: Google Colab T4 GPU (~10 minutes)

## Built for
CIS433 — Deep Learning for Text Analytics  
Simon Business School, University of Rochester

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo structure
```
├── app.py                    # Streamlit app
├── rui_torch_transformer.py  # Transformer architecture
├── requirements.txt
├── saved_model/
│   ├── translator_weights.pt
│   ├── src_vocab.pkl
│   ├── tgt_vocab.pkl
│   └── config.pkl
└── README.md
```
