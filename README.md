# 🌍 AI Live Text Translator

A multilingual AI-powered text translation application built with **Python, PyTorch, Hugging Face Transformers, NLLB-200, and Gradio**.

The project provides **live text translation with automatic language detection**, allowing users to type text and receive a translation automatically after a short delay.

---

## 🚀 Project Overview

The **AI Live Text Translator** uses Meta's **NLLB-200 (No Language Left Behind)** multilingual neural machine translation model to translate text between multiple languages.

Unlike a simple translation API wrapper, this project runs a pretrained neural machine translation model locally and integrates it into an interactive Gradio application.

The application was developed and tested using **Google Colab with GPU acceleration**.

---

## ✨ Features

- 🌍 Multilingual neural machine translation
- 🔍 Automatic language detection
- ⚡ Live translation while typing
- ⏱️ Debounced translation using a 1-second timer
- 🔄 Swap source and target languages
- 🌐 Manual translation option
- 🗑️ Clear input and output
- 🤖 NLLB-200 multilingual translation model
- 🚀 GPU-accelerated inference
- 🖥️ Interactive Gradio web interface

---

## 🌐 Supported Languages

Currently supported:

| Language | NLLB Language Code |
|---|---|
| 🇬🇧 English | `eng_Latn` |
| 🇩🇪 German | `deu_Latn` |
| 🇫🇷 French | `fra_Latn` |
| 🇪🇸 Spanish | `spa_Latn` |
| 🇮🇹 Italian | `ita_Latn` |
| 🇮🇳 Hindi | `hin_Deva` |

The project can be extended to support additional languages available in NLLB-200.

---

## 🧠 AI Model

This project uses:

**Model:** `facebook/nllb-200-distilled-600M`

NLLB-200 is a multilingual neural machine translation model designed to support translation across a large number of languages.

The model is loaded using the Hugging Face Transformers library and performs sequence-to-sequence translation using PyTorch.

---

## 🏗️ System Architecture

```text
User Input
    │
    ▼
Gradio Interface
    │
    ▼
1-Second Timer / Manual Translation
    │
    ▼
Language Detection
    │
    ▼
Source Language Identification
    │
    ▼
NLLB-200 Transformer Model
    │
    ▼
GPU-Accelerated Inference
    │
    ▼
Translated Text
    │
    ▼
Gradio Output
