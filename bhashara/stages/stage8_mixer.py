"""
Bhashara Stage 8: Audio Mixing & Mastering
Mixes dubbed voice stem over preserved instrumental track with sidechain ducking,
crossfading, and EBU R128 loudness normalization (-16 LUFS).
CPU-bound stage.
"""

import logging
import numpy as np
from pathlib import Path
from bhashara.models.schema import ProjectMetadata
from bhashara.utils.audio_utils import (
    load_audio,
    save_audio,
    sidechain_duck,
    normalize_loudness_lufs
)
from bhashara.config import TARGET_LOUDNESS_LUFS, DUCKING_ATTENUATION_DB

logger = logging.getLogger("bhashara.stage8")


def run_stage8_mixer(project: ProjectMetadata, project_dir: Path) -> ProjectMetadata:
    """
    Assembles dialogue timeline, applies sidechain ducking on BGM, and master normalizes.
    Resumable: Skips if master_dubbed_audio already exists and stage completed.
    """
    stems_dir = project_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    
    dialogue_stem_path = stems_dir / "dubbed_dialogue.wav"
    master_mix_path = stems_dir / "master_dubbed_audio.wav"

    if (
        project.stage_status.get("mixer") == "completed"
        and master_mix_path.exists()
        and not any(s.dirty for s in project.segments)
    ):
        logger.info("[Stage 8] Mix already completed. Resuming.")
        project.dubbed_dialogue_stem_path = str(dialogue_stem_path)
        project.final_dubbed_audio_path = str(master_mix_path)
        return project

    logger.info("[Stage 8] Starting stems mixing, ducking, and -16 LUFS mastering...")
    
    # 1. Load instrumental stem (or base audio)
    bgm_path = Path(project.instrumental_stem_path or project.audio_16k_mono)
    bgm_audio, sr = load_audio(bgm_path)
    total_samples = len(bgm_audio)
    
    # 2. Build Dialogue Timeline Track
    dialogue_track = np.zeros(total_samples, dtype=np.float32)
    
    for seg in project.segments:
        if not seg.fitted_audio_path or not Path(seg.fitted_audio_path).exists():
            continue
            
        seg_audio, seg_sr = load_audio(Path(seg.fitted_audio_path))
        start_samp = int(seg.start * sr)
        end_samp = min(total_samples, start_samp + len(seg_audio))
        
        insert_len = end_samp - start_samp
        if insert_len > 0:
            # Crossfade smoothing (10ms)
            fade_samples = min(int(sr * 0.01), insert_len // 4)
            insert_data = seg_audio[:insert_len].copy()
            if fade_samples > 0:
                insert_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
                insert_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
            dialogue_track[start_samp:end_samp] += insert_data

    # Save isolated dialogue stem
    save_audio(dialogue_stem_path, dialogue_track, sr)

    # 3. Dynamic Sidechain Ducking on BGM
    logger.info(f"[Stage 8] Applying sidechain ducking ({DUCKING_ATTENUATION_DB} dB)...")
    ducked_bgm = sidechain_duck(
        dialogue=dialogue_track,
        bgm=bgm_audio,
        sr=sr,
        duck_attenuation_db=DUCKING_ATTENUATION_DB,
        attack_ms=50,
        release_ms=250
    )

    # 4. Master Summing & EBU R128 Normalization
    logger.info(f"[Stage 8] Normalizing loudness to target {TARGET_LOUDNESS_LUFS} LUFS...")
    master_mix = dialogue_track * 1.1 + ducked_bgm * 0.9
    master_normalized = normalize_loudness_lufs(master_mix, target_lufs=TARGET_LOUDNESS_LUFS)
    
    save_audio(master_mix_path, master_normalized, sr)

    project.dubbed_dialogue_stem_path = str(dialogue_stem_path)
    project.final_dubbed_audio_path = str(master_mix_path)
    project.stage_status["mixer"] = "completed"
    project.save(project_dir / "segments.json")
    
    logger.info("[Stage 8] Audio mixing and mastering complete.")
    return project
