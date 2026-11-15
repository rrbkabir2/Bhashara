"""
Bhashara Configuration Module
Centralized settings, directories, audio targets, and model configs.
"""

from pathlib import Path
import os

try:
    import torch
    DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
except ImportError:
    torch = None
    DEVICE = "cpu"

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Audio Standards
AUDIO_SAMPLE_RATE = 16000     # 16kHz for ASR, Diarization, and VAD
MIX_SAMPLE_RATE = 44100       # 44.1kHz standard for final video muxing
TARGET_LOUDNESS_LUFS = -16.0  # EBU R128 broadcast / YouTube standard
TRUE_PEAK_DBTP = -1.0         # True peak headroom
DUCKING_ATTENUATION_DB = -10.0 # BGM attenuation when dialogue active
DUCKING_ATTACK_MS = 50        # Ducking fade-in
DUCKING_RELEASE_MS = 250      # Ducking recovery

# Timing and Time-Stretch Constraints
MAX_SPEEDUP_FACTOR = 1.15     # Max acoustic speedup without chipmunk distortion
MIN_SPEEDDOWN_FACTOR = 0.90   # Max acoustic slowdown
HINDI_SYLLABLES_PER_SEC = 4.0 # Conversational rate for Hindi length budgeting
ENGLISH_SYLLABLES_PER_SEC = 4.2

# Hardware & VRAM Manager
MIN_FREE_VRAM_GB = 1.8        # Fallback to CPU if free VRAM drops below this
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

# Concurrency & Overlap Queue
CPU_WORKERS = max(1, os.cpu_count() - 1 if os.cpu_count() else 2)
MAX_OVERLAP_QUEUE_SIZE = 4    # Small bounded queue to prevent RAM exhaustion

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi"
}
