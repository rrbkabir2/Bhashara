"""
Bhashara Stage 7: Timing Fit & Anti-Collision Alignment
Shortens pauses, applies pitch-preserving time-stretching (capped at 10-15%),
and enforces strict non-overlapping dialogue constraints.
CPU-bound stage.
"""

import logging
import numpy as np
from pathlib import Path
from bhashara.models.schema import ProjectMetadata
from bhashara.utils.audio_utils import (
    load_audio,
    save_audio,
    compress_pauses,
    time_stretch_sola
)
from bhashara.config import MAX_SPEEDUP_FACTOR, MIN_SPEEDDOWN_FACTOR

logger = logging.getLogger("bhashara.stage7")


def run_stage7_timing_fit(project: ProjectMetadata, project_dir: Path) -> ProjectMetadata:
    """
    Fits synthesized audio clips into original timing slots without overlapping.
    Resumable: Skips segments that already have fitted_audio_path and are completed.
    """
    aligned_dir = project_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[Stage 7] Starting 3-tier timing fit & anti-collision alignment...")
    
    num_segments = len(project.segments)
    
    for i, seg in enumerate(project.segments):
        out_wav = aligned_dir / f"seg_{seg.id:04d}.wav"
        
        # Check if already fitted and clean
        if not seg.dirty and out_wav.exists() and seg.status == "completed":
            seg.fitted_audio_path = str(out_wav)
            continue
            
        if not seg.raw_tts_path or not Path(seg.raw_tts_path).exists():
            logger.warning(f"[Stage 7] Seg #{seg.id} missing raw TTS audio. Skipping.")
            continue
            
        audio, sr = load_audio(Path(seg.raw_tts_path))
        curr_dur = len(audio) / sr
        target_dur = seg.target_duration
        
        # Next segment boundary for anti-collision
        next_start = project.segments[i + 1].start if i + 1 < num_segments else seg.end + 5.0
        max_allowed_dur = max(0.5, next_start - seg.start - 0.05)  # 50ms safety cushion
        
        logger.info(f"[Stage 7] Seg #{seg.id}: Current={curr_dur:.2f}s, Target={target_dur:.2f}s (Max={max_allowed_dur:.2f}s)")

        # Tier 1: Micro-pause compression
        if curr_dur > target_dur:
            audio = compress_pauses(audio, sr, max_silence_sec=0.08)
            curr_dur = len(audio) / sr

        # Tier 2: Pitch-preserving SOLA time stretch (capped at 10-15%)
        speed_ratio = curr_dur / target_dur
        clamped_speed = np.clip(speed_ratio, MIN_SPEEDDOWN_FACTOR, MAX_SPEEDUP_FACTOR)
        
        if abs(clamped_speed - 1.0) > 0.02:
            audio = time_stretch_sola(audio, sr, float(clamped_speed))
            curr_dur = len(audio) / sr
            seg.speed_factor = round(float(clamped_speed), 2)

        # Tier 3: Strict anti-collision trimming / padding
        if curr_dur > max_allowed_dur:
            # Gentle fade-out before hard cap
            max_samples = int(max_allowed_dur * sr)
            fade_samples = int(sr * 0.05)
            audio = audio[:max_samples]
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        elif curr_dur < target_dur * 0.8:
            # Pad slight trailing silence up to target
            target_samples = int(target_dur * sr)
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            audio = padded

        save_audio(out_wav, audio, sr)
        seg.fitted_audio_path = str(out_wav)
        seg.status = "completed"
        logger.info(f"[Stage 7] Seg #{seg.id} fitted -> final duration: {len(audio)/sr:.2f}s")

    project.stage_status["timing"] = "completed"
    project.save(project_dir / "segments.json")
    
    logger.info("[Stage 7] Timing fit stage complete.")
    return project
