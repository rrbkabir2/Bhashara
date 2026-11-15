# Bhashara - Licenses & Open Source Attribution

Bhashara is committed to being 100% free, open-source, and locally runnable. Every model, library, and tool utilized in the pipeline has its license detailed below.

---

## 1. Audio Separation
* **Tool / Model**: [Demucs v4 (Hybrid Transformer Demucs / htdemucs)](https://github.com/facebookresearch/demucs)
* **Author**: Meta AI Research (Alexandre Défossez et al.)
* **License**: **MIT License**
* **Usage**: Separation of dialogue vocals from background music and sound effects stems. Completely free for open-source and commercial use.

---

## 2. Speech-to-Text & Alignment
* **Tool / Model**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (powered by [CTranslate2](https://github.com/OpenNMT/CTranslate2) and OpenAI Whisper weights)
* **Authors**: SYSTRAN & OpenAI
* **License**: **MIT License** (faster-whisper & CTranslate2), **Apache License 2.0** (OpenAI Whisper weights)
* **Usage**: High-speed, low-VRAM speech transcription with word-level timestamps and source language detection.

---

## 3. Speaker Diarization
* **Tool / Model**: [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)
* **Author**: Hervé Bredin et al.
* **License**: **MIT License** (Code). Note: Hugging Face model weights (`pyannote/speaker-diarization-3.1`) require accepting user terms on Hugging Face.
* **Fallback / Offline Diarizer**: WhisperX / Acoustic clustering (`librosa` + `scipy` cosine distance).
* **License**: **BSD 2-Clause / BSD 3-Clause**.

---

## 4. Translation & Length Budgeting
* **Tool / Model**: Contextual LLM / Offline MarianMT / Local Ollama / vLLM / HuggingFace Transformers
* **Models**:
  - [Helsinki-NLP/opus-mt-en-hi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi) & `opus-mt-hi-en`: **CC-BY-4.0 / Apache 2.0**
  - Local LLM prompt adapters (Ollama / Llama-3 / Mistral / Qwen-2.5): **Apache 2.0 / Llama 3 Community License**
* **Usage**: Dialogue translation with strict syllable and time constraints.

---

## 5. Voice Cloning & Speech Synthesis (TTS)
* **Tool / Model 1 (Primary Cloner)**: [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
  * **License**: **Coqui Public Model License (CPML)** - Free for personal, non-commercial, and research use.
* **Tool / Model 2 (Fast Zero-VRAM Baseline)**: [Edge-TTS](https://github.com/rany2/edge-tts)
  * **License**: **MIT License** (library). Unofficial interface to public edge neural endpoints.
* **Tool / Model 3 (Swappable Open Weights)**: [F5-TTS](https://github.com/SWivid/F5-TTS) / [Chatterbox](https://github.com/resemble-ai)
  * **License**: **MIT / Apache 2.0**.

---

## 6. Time Stretching & Audio Processing
* **Tool / Library**: [Librosa](https://github.com/librosa/librosa), [SoundFile](https://github.com/bastibe/python-soundfile), [SciPy](https://github.com/scipy/scipy), [Rubberband-cli](https://breakfastquay.com/rubberband/)
* **License**: **ISC License** (Librosa), **BSD 3-Clause** (Soundfile & Scipy), **GPL v2+** (Rubberband).
* **Usage**: Pitch-preserved time compression, micro-pause truncation, anti-collision padding.

---

## 7. Media Processing & Ingestion
* **Tool**: [FFmpeg](https://ffmpeg.org/)
* **License**: **LGPL v2.1+ / GPL v2+** (depending on build flags).
* **Tool**: [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* **License**: **The Unlicense** (Public Domain).

---

## 8. Web Framework & Runtime
* **Pydantic**: **MIT License**
* **Gradio**: **Apache License 2.0**
* **PyTorch**: **Modified BSD License**

---
*Verified by Bhashara Engineering Team. All dependencies are non-proprietary and run 100% locally on personal hardware.*
