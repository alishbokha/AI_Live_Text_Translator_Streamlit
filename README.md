# 🌍 AI Live Text Translator

A multilingual AI text translation application using **NLLB-200**, Hugging Face Transformers, PyTorch and Streamlit.

## Improvements in this version

- Cached model loading with `st.cache_resource`
- Faster interactive generation using greedy decoding (`num_beams=1`)
- `torch.inference_mode()` for inference
- Shorter generation limits for interactive use
- Correct language swap logic
- Automatic language detection
- Translation only when the input/settings change

## Model

`facebook/nllb-200-distilled-600M`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
