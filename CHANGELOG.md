# Bhashara Changelog

All notable changes to the Bhashara AI Video Dubbing Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - While GPU generates TTS for segment K, CPU concurrently executes SOLA time-stretching, pause compression, and alignment for segment K-1.
  - Bounded queue prevents RAM inflation on consumer hardware.
- **Milestone 1 Verification (10-Second Clip)**:
  - Validated end-to-end on synthetic multi-speaker media (Male + Female dialogue with BGM).
  - Diarization successfully identified distinct speakers with F0 acoustic pitch classification.
  - Granular single-line re-dubbing executed in 0.16 seconds without re-running other segments.
  - Measured 8.3% speedup on 10s clip with overlap enabled.
- **Web Interface**:
  - Standalone local web UI featuring input submission, stage-by-stage progress monitoring, side-by-side script editor, and segment preview playback.
- **Documentation**:
  - `LICENSES.md`: Open-source licensing manifest covering every model and library.
