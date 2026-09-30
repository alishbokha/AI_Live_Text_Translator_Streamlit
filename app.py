import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

st.set_page_config(
    page_title="AI Live Text Translator",
    page_icon="🌍",
    layout="wide"
)

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


@st.cache_resource(show_spinner="Loading AI translation model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    # CPU inference can benefit from a moderate number of threads.
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    return tokenizer, model, device


tokenizer, model, device = load_model()


def detect_language(text):
    if not text or not text.strip():
        return None

    try:
        code = detect(text)
        return LANGUAGE_DETECT_MAP.get(code)
    except Exception:
        return None


def translate_text(text, source_language, target_language):
    if not text or not text.strip():
        return ""

    if source_language == target_language:
        return text

    source_code = LANGUAGE_MAP[source_language]
    target_code = LANGUAGE_MAP[target_language]

    tokenizer.src_lang = source_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    target_token_id = tokenizer.convert_tokens_to_ids(target_code)

    # Faster than beam-search generation and appropriate for interactive translation.
    with torch.inference_mode():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )

    return tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]


def get_effective_source(text, selected_source, auto_detect):
    if auto_detect:
        detected = detect_language(text)
        if detected in LANGUAGE_MAP:
            return detected, detected
    return selected_source, None


# Session state
defaults = {
    "source_language": "English",
    "target_language": "German",
    "output_text": "",
    "last_translated_input": "",
    "last_source": "",
    "last_target": "",
    "last_auto_detect": True,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("🌍 AI Live Text Translator")
st.markdown(
    "Multilingual neural machine translation powered by "
    "**NLLB-200** and Hugging Face Transformers."
)

st.info(
    "Supported languages: English, German, French, Spanish, Italian and Hindi."
)

# Language selection
col1, col2, col3 = st.columns([4, 1, 4])

with col1:
    source_language = st.selectbox(
        "From",
        list(LANGUAGE_MAP.keys()),
        index=list(LANGUAGE_MAP.keys()).index(
            st.session_state.source_language
        ),
        key="source_language_widget",
    )

with col2:
    st.write("")
    st.write("")
    if st.button("⇄ Swap", use_container_width=True):
        old_source = st.session_state.source_language
        old_target = st.session_state.target_language

        st.session_state.source_language = old_target
        st.session_state.target_language = old_source

        st.session_state.source_language_widget = old_target
        st.session_state.target_language_widget = old_source

        st.rerun()

with col3:
    target_language = st.selectbox(
        "To",
        list(LANGUAGE_MAP.keys()),
        index=list(LANGUAGE_MAP.keys()).index(
            st.session_state.target_language
        ),
        key="target_language_widget",
    )

st.session_state.source_language = source_language
st.session_state.target_language = target_language

auto_detect = st.checkbox(
    "🔎 Automatically detect input language",
    value=st.session_state.last_auto_detect,
)

# Input
input_text = st.text_area(
    "Input",
    height=220,
    placeholder="Type or paste text here...",
    key="input_text",
)

# Automatic live translation:
# Streamlit reruns when the user interacts with the app. This translates
# when the current input differs from the last translated input.
# For true timed keystroke-by-keystroke updates, a custom frontend component
# would be needed; this version avoids sending a model request for every keystroke.
if input_text.strip():
    effective_source, detected_language = get_effective_source(
        input_text,
        source_language,
        auto_detect,
    )

    settings_changed = (
        input_text != st.session_state.last_translated_input
        or effective_source != st.session_state.last_source
        or target_language != st.session_state.last_target
        or auto_detect != st.session_state.last_auto_detect
    )

    if settings_changed:
        with st.spinner("Translating..."):
            try:
                st.session_state.output_text = translate_text(
                    input_text,
                    effective_source,
                    target_language,
                )
                st.session_state.last_translated_input = input_text
                st.session_state.last_source = effective_source
                st.session_state.last_target = target_language
                st.session_state.last_auto_detect = auto_detect
            except Exception as e:
                st.session_state.output_text = ""
                st.error(f"Translation error: {e}")

    if auto_detect and detected_language:
        st.caption(f"🔎 Detected language: **{detected_language}**")
else:
    st.session_state.output_text = ""
    st.session_state.last_translated_input = ""


# Output
st.subheader("Translation")
st.text_area(
    "Translation",
    value=st.session_state.output_text,
    height=220,
    placeholder="Your translation will appear here...",
    label_visibility="collapsed",
)


# Manual controls
button_col1, button_col2 = st.columns(2)

with button_col1:
    if st.button("🌐 Translate", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to translate.")
        else:
            effective_source, detected_language = get_effective_source(
                input_text,
                source_language,
                auto_detect,
            )
            with st.spinner("Translating..."):
                try:
                    st.session_state.output_text = translate_text(
                        input_text,
                        effective_source,
                        target_language,
                    )
                    st.session_state.last_translated_input = input_text
                    st.session_state.last_source = effective_source
                    st.session_state.last_target = target_language
                    st.session_state.last_auto_detect = auto_detect
                    st.rerun()
                except Exception as e:
                    st.error(f"Translation error: {e}")

with button_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        for key in [
            "output_text",
            "last_translated_input",
            "last_source",
            "last_target",
        ]:
            st.session_state[key] = ""
        st.rerun()


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
        - Interactive translation
        - Cached model loading for faster subsequent requests
        """
    )

st.caption(f"Running on: {device} | Model: {MODEL_NAME}")
