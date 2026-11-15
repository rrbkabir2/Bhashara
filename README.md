# 🎙️ Bhashara (भशारा) - Local AI Video Dubbing Engine

**Bhashara** is a 100% free, open-source, and locally executed AI video dubbing pipeline engineered for consumer GPUs (6 GB VRAM) with seamless CPU fallback. It dubs videos into Hindi or English while preserving each character's distinct voice, maintaining precise time synchronization, and keeping background music and sound effects intact.

---

## ⚡ Core Engineering Highlights

1. **Strict 1-Model VRAM Janitor:**
   - Sequential model execution with proactive memory reclamation (`gc.collect()` + `torch.cuda.empty_cache()`).
   - Consumer GPU protection: Automatically offloads stages to CPU if free VRAM drops below 1.8 GB.
2. **CPU & GPU Overlap Pipeline:**
   - Asynchronous producer-consumer queue between GPU and CPU.
   - While GPU executes TTS speech synthesis on Segment $K$, CPU executes time-stretching, pause compression, and audio alignment on Segment $K-1$.
   - Bounded queue prevents RAM inflation.
3. **Dual-Engine Diarizer with Zero-Login Fallback:**
   - Built-in acoustic pitch ($F_0$) gender clustering and reference audio curation without requiring Hugging Face tokens.
   - Pro Pyannote 3.1 support with optional `HF_TOKEN`.
4. **3-Tier Syllable-Budgeted Timing Fit:**
   - **Tier 1:** Syllable budgeting during translation ($S_{\max} = \lfloor T \times 4.0\text{ syl/s} \rfloor$) to prevent Hindi expansion blowouts.
   - **Tier 2:** Micro-pause contraction ($> 150\text{ ms} \rightarrow 60\text{ ms}$).
   - **Tier 3:** Pitch-preserved WSOLA / SOLA time stretching capped at $10\text{–}15\%$ with anti-collision padding.
5. **Master Audio Mixing & Mastering:**
   - Demucs 2-stem vocal/instrumental separation.
   - Dynamic sidechain ducking ($-10\text{ dB}$) with 50ms attack and 250ms release.
   - EBU R128 loudness normalization to **$-16\text{ LUFS}$** with $-1.0\text{ dBTP}$ ceiling.
6. **Granular Single-Line Re-dubbing:**
   - `segments.json` acts as the single source of truth.
   - Modifying a single line regenerates only that segment and remuxes in seconds without re-running the full video.

---

## 🏗️ 10-Stage Pipeline Architecture

| Stage | Name | Compute | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Input Ingestion** | CPU | Local video or YouTube URL (`yt-dlp`), 16kHz mono audio extraction. |
| **2** | **Stem Separation** | GPU | Demucs (`htdemucs`) separates vocals from BGM and sound effects. |
| **3** | **Transcription** | GPU | `faster-whisper` with word-level timestamps and VAD filtering. |
| **4** | **Diarization** | GPU/CPU | Speaker clustering, acoustic $F_0$ gender detection, reference clip extraction. |
| **5** | **Translation** | CPU | Syllable-budgeted translation with Devanagari text normalization. |
| **6** | **Voice Synthesis** | GPU/CPU | XTTS-v2 zero-shot cloning, Edge-TTS, and local acoustic preset adapters. |
| **7** | **Timing Fit** | CPU | 3-tier duration fitting, time stretch ($\le 1.15\times$), and anti-collision. |
| **8** | **Mixer & Mastering**| CPU | Sidechain ducking ($-10\text{ dB}$), crossfades, and $-16\text{ LUFS}$ normalization. |
| **9** | **Output Packaging**| CPU | Remuxes MP4/MKV and compiles original, translated, and bilingual subtitles (`.srt`). |
| **10**| **Lip Sync Hook** | GPU | Modular hook for Wav2Lip / LatentSync (camera-shot chunked, off by default). |

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- FFmpeg installed and in PATH
- Optional: NVIDIA GPU with CUDA 11.8+ / 12.x for accelerated inference

### 2. Installation
```bash
# Clone repository
git clone https://github.com/rrbkabir2/Bhashara.git
cd Bhashara

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Test Suite (Milestone 1)
```bash
python tests/test_10s_clip.py
```

### 4. Launching the Web UI Studio
```bash
python ui/app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 📜 Open Source Licenses
All libraries, models, and dependencies are 100% open-source and free for local use. See [LICENSES.md](LICENSES.md) for full details.
