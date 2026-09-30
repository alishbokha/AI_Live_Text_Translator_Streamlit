import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory

# Make language detection reproducible
DetectorFactory.seed = 0

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Live Text Translator",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------
# Supported languages
# ---------------------------------------------------------
LANGUAGE_MAP = {
    "English": "eng_Latn",
    "German": "deu_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Hindi": "hin_Deva",
}

LANGUAGE_DETECT_MAP = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "hi": "Hindi",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"


# ---------------------------------------------------------
# Load model only once
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI translation model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# ---------------------------------------------------------
# Translation
# ---------------------------------------------------------
def translate_text(text, source_language, target_language):
    if not text or not text.strip():
        return ""

    if source_language not in LANGUAGE_MAP:
        raise ValueError(f"Unsupported source language: {source_language}")

    if target_language not in LANGUAGE_MAP:
        raise ValueError(f"Unsupported target language: {target_language}")

    if source_language == target_language:
        return text

    source_code = LANGUAGE_MAP[source_language]
    target_code = LANGUAGE_MAP[target_language]

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    target_token_id = tokenizer.convert_tokens_to_ids(target_code)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_length=512
        )

    return tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]


# ---------------------------------------------------------
# Language detection
# ---------------------------------------------------------
def detect_language(text):
    if not text or not text.strip():
        return None

    try:
        detected_code = detect(text)
        return LANGUAGE_DETECT_MAP.get(detected_code)
    except Exception:
        return None


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🌍 AI Live Text Translator")
st.markdown(
    "Translate text between multiple languages using "
    "**NLLB-200**, a multilingual neural machine translation model."
)

st.info(
    "Supported languages: English, German, French, Spanish, Italian and Hindi."
)

# Initialize session state
if "source_language" not in st.session_state:
    st.session_state.source_language = "English"

if "target_language" not in st.session_state:
    st.session_state.target_language = "German"

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "output_text" not in st.session_state:
    st.session_state.output_text = ""

if "auto_detect" not in st.session_state:
    st.session_state.auto_detect = True


# ---------------------------------------------------------
# Language controls
# ---------------------------------------------------------
col1, col2, col3 = st.columns([4, 1, 4])

with col1:
    source_language = st.selectbox(
        "From",
        list(LANGUAGE_MAP.keys()),
        index=list(LANGUAGE_MAP.keys()).index(
            st.session_state.source_language
        ),
        key="source_select"
    )

with col2:
    st.write("")
    st.write("")
    if st.button("⇄ Swap", use_container_width=True):
        st.session_state.source_language = st.session_state.target_language
        st.session_state.target_language = st.session_state.source_language
        st.rerun()

with col3:
    target_language = st.selectbox(
        "To",
        list(LANGUAGE_MAP.keys()),
        index=list(LANGUAGE_MAP.keys()).index(
            st.session_state.target_language
        ),
        key="target_select"
    )

# Keep selected languages synchronized
st.session_state.source_language = source_language
st.session_state.target_language = target_language

auto_detect = st.checkbox(
    "🔎 Automatically detect input language",
    value=st.session_state.auto_detect
)
st.session_state.auto_detect = auto_detect


# ---------------------------------------------------------
# Text areas
# ---------------------------------------------------------
input_col, output_col = st.columns(2)

with input_col:
    st.subheader("Input")

    input_text = st.text_area(
        "Enter text",
        value=st.session_state.input_text,
        height=220,
        placeholder="Type or paste text here...",
        label_visibility="collapsed"
    )

with output_col:
    st.subheader("Translation")

    output_placeholder = st.empty()

    if st.session_state.output_text:
        output_placeholder.text_area(
            "Translated text",
            value=st.session_state.output_text,
            height=220,
            label_visibility="collapsed"
        )
    else:
        output_placeholder.text_area(
            "Translated text",
            value="",
            height=220,
            placeholder="Your translation will appear here...",
            label_visibility="collapsed"
        )


# ---------------------------------------------------------
# Action buttons
# ---------------------------------------------------------
button_col1, button_col2 = st.columns(2)

with button_col1:
    translate_clicked = st.button(
        "🌐 Translate",
        type="primary",
        use_container_width=True
    )

with button_col2:
    clear_clicked = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# ---------------------------------------------------------
# Clear
# ---------------------------------------------------------
if clear_clicked:
    st.session_state.input_text = ""
    st.session_state.output_text = ""
    st.rerun()


# ---------------------------------------------------------
# Translate
# ---------------------------------------------------------
if translate_clicked:
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    else:
        detected_language = None

        if auto_detect:
            detected_language = detect_language(input_text)

            if detected_language is not None:
                st.session_state.source_language = detected_language

        effective_source_language = (
            detected_language
            if detected_language in LANGUAGE_MAP
            else source_language
        )

        try:
            with st.spinner("Translating..."):
                translation = translate_text(
                    input_text,
                    effective_source_language,
                    target_language
                )

            st.session_state.input_text = input_text
            st.session_state.output_text = translation

            if detected_language:
                st.success(
                    f"Detected language: {detected_language}"
                )

            st.rerun()

        except Exception as e:
            st.error(f"Translation error: {e}")


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
with st.expander("ℹ️ About this project"):
    st.write(
        """
        **Model:** facebook/nllb-200-distilled-600M

        **Frameworks:** Hugging Face Transformers, PyTorch, Streamlit

        **Features:**
        - Multilingual neural machine translation
        - Automatic language detection
        - Manual source/target language selection
        - Language swapping
        - Translation history for the current session
        """
    )

st.caption(
    f"Running on: {device} | Model: {MODEL_NAME}"
)
