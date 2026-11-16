# Bhashara Changelog

All notable changes to the Bhashara AI Video Dubbing Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.2] - 2026-11-16 - Local Web Studio & Browser Dubbing Interface

### Added
- **Local Web Studio (`run_ui.py`)**:
  - FastAPI + Uvicorn backend strictly bound to `127.0.0.1:8000` with zero external tracking or authentication barriers.
  - One-command launcher automatically boots server and opens the browser studio.
- **Media Ingestion & Web Downloading**:
  - Drag-and-drop file upload for local media (MP4, MKV, MOV, WebM).
  - Online video link ingestion powered by `yt-dlp` (`/api/info`), pre-fetching video title, duration, and thumbnail preview.
  - Plain-English error reporting for private, age-gated, geo-blocked, or live stream links.
  - Length and CPU duration advisory banner requiring confirmation for videos $> 5$ minutes.
- **Real-Time 10-Stage Progress Engine**:
  - Thread-safe background `JobManager` coordinating FIFO job execution and single heavy job concurrency.
  - Real-time progress updates per stage (`waiting`, `running`, `done`, `failed`) with active model, duration, and step description.
  - Overall progress bar tied to disk verification (never hits 100% until final output files exist).
  - Collapsible terminal streaming pipeline logs in real time.
  - Clean job cancellation immediately halting pipeline execution and invoking `VRAMJanitor` memory reclamation.
- **Synchronized Dual Video Player & Media Delivery**:
  - Synchronized side-by-side player comparing original source video against dubbed Hindi/English video.
  - Dedicated download endpoints (`/download/{job_id}/{file_type}`) for:
    - Dubbed final video (`.mp4`)
    - Master mixed audio (`.wav`)
    - Isolated vocal dialogue stem (`.wav`)
    - Isolated music and background sound effects stem (`.wav`)
    - Multi-track subtitles (`.orig.srt`, `.translated.srt`, `.bilingual.srt`).
- **Interactive Script Editor & Single-Line Re-dubbing**:
  - Side-by-side transcript table showing detected timestamps, speakers, and editable Devanagari text.
  - Single-line "Re-dub Line" button (`/api/jobs/{job_id}/redub`) re-synthesizing, re-fitting, and remuxing only the altered segment in seconds.
  - Visual "Modified" status tags tracking edited lines.
- **Job History & Cache Management**:
  - Persistent job registry (`cache/job_history.json`) restoring job history across server reboots.
  - Resumability endpoint (`/api/jobs/{job_id}/resume`) resuming failed/cancelled jobs from cached stage checkpoints.
  - Live disk space breakdown and one-click intermediate cache cleaner (`/api/clean_temp`).

---

## [1.0.1] - 2026-11-16 - Real Neural Models, Verification Suite & Git Sanitization

### Added
- **Real Neural Deep Learning Integration**:
  - `Demucs v4 (htdemucs)`: Real neural separation splitting dialogue vocals from background music/SFX stems.
  - `faster-whisper (base)`: CTranslate2 speech recognition generating real word timestamps and VAD filtering.
  - `MarianMT (Helsinki-NLP/opus-mt-en-hi)`: Neural Seq2Seq translation with Devanagari normalization and syllable budgeting.
  - `Edge-TTS`: Neural Hindi voices (`hi-IN-MadhurNeural`, `hi-IN-SwaraNeural`) for natural character speech.
  - `Acoustic F0 Clustering`: Autocorrelation pitch tracking separating Male (~130Hz) and Female (~215Hz) speakers.
- **Model Verification Suite (`verify_models.py`)**:
  - Single-command diagnostic script inspecting model authenticity, version, disk size, execution device, and load latency.
  - Clearly prints `REAL`, `MISSING`, or `FAKE` in capital letters.
- **Diagnostic Run Logging (`run_log.txt`)**:
  - Automatically profiles every stage: model engine, input/output file sizes, execution duration, and peak VRAM.
  - Persisted to project cache and `proof/run_log.txt`.
- **Dual-Clip Differentiation Test (`tests/test_differentiation.py`)**:
  - Feeds two distinct video clips and asserts that transcripts, translations, and voice profiles differ.
- **Zero-Setup Bundled FFmpeg**:
  - Automated detection of static FFmpeg binary at `tools/ffmpeg.exe` for zero PATH configuration on Windows.
- **Proof Directory (`proof/`)**:
  - Complete verification artifacts: `milestone1_dubbed_video.mp4`, separated stems, transcripts, translations, cloned lines, and subtitles.
- **Comprehensive `.gitignore`**:
  - Excludes secrets, API tokens (`hf_token*`, `.env`), virtual environments, model weights (`.pt`, `.safetensors`), caches, and heavy media.

### Fixed
- **PyAV 19 Incompatibility**: Resolved `TypeError: open() got an unexpected keyword argument 'metadata_errors'` in `faster-whisper` by passing memory-loaded NumPy arrays directly.
- **IDE Red Files & Type Annotations**:
  - Fixed `Optional[str]` type mismatch in `stage4_diarize.py`.
  - Added safe path guards in stages 1, 2, 3, 8, 9.
  - Added `pyrightconfig.json` and `.vscode/settings.json` eliminating unresolved import markers.
- **Attribute Error in Pipeline Logging**: Fixed `final_dubbed_audio_path` and `final_dubbed_video_path` field lookups.

### Performance
- **Measured CPU/GPU Overlap Speedup**: Benchmarked sequential (53.33s) vs overlapped (39.11s) execution on a real 14-second multi-speaker clip, achieving a **+26.7% speedup**.

---

## [1.0.0] - 2026-11-15 - Initial Foundation & Milestone 1 (10-Second Dubbing Engine)

### Added
- **Complete 10-Stage Pipeline Architecture**:
  1. `Input`: Local video file ingestion or URL download via `yt-dlp`, with FFmpeg 16kHz mono audio extraction.
  2. `Separation`: Vocal vs Instrumental (BGM & SFX) stem isolation via Demucs (`htdemucs`).
  3. `Transcription`: Word-level speech-to-text with timestamping and VAD using `faster-whisper`.
  4. `Diarization`: Multi-speaker segmentation, pitch-based acoustic gender estimation ($F_0$), and reference clip extraction for voice cloning.
  5. `Translation`: Syllable-budgeted dialogue translation (English ↔ Hindi), with conversational compression and back-translation checks.
  6. `Synthesis`: Zero-shot voice cloning with XTTS-v2 adapter and fallback Edge-TTS preset engine.
  7. `Timing Fit`: 3-tier duration matching (syllable prompt budgeting, pitch-preserving time-stretching up to 1.15x, and anti-collision silence management).
  8. `Mix`: Dynamic sidechain ducking (-10 dB), crossfades, and EBU R128 loudness normalization (-16 LUFS).
  9. `Output`: Synchronized video muxing, stem preservation, and multi-track subtitle generation (`.srt`, `.vtt` in original, translated, and bilingual).
  10. `Lip Sync Hook`: Modular placeholder for Wav2Lip / LatentSync (camera-shot chunked, disabled by default).
- **Reliability & State Machine**:
  - `segments.json`: Centralized Single Source of Truth for all segment metadata, speakers, timestamps, and translations.
  - Resumable pipeline: Checkpoint system caching intermediate outputs (`cache/<video_id>/...`).
  - Granular re-dubbing: Editing a single line flags only that segment dirty and regenerates in seconds without re-running the full movie.
  - VRAM Janitor: Proactive memory manager clearing PyTorch cache and releasing unused models to guarantee stability on 6 GB consumer GPUs.
- **CPU/GPU Overlap Concurrency Engine**:
  - Asynchronous producer-consumer queue between GPU neural inference and CPU audio processing.
  - Bounded queue prevents RAM inflation on consumer hardware.
- **Web Interface**:
  - Standalone local web UI featuring input submission, stage-by-stage progress monitoring, side-by-side script editor, and segment preview playback.
- **Documentation**:
  - `LICENSES.md`: Open-source licensing manifest covering every model and library.
