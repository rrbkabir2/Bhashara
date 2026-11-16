"""
Bhashara Stage 2: Audio Separation (Demucs v4)
Splits dialogue vocals from background music and sound effects stems using Demucs htdemucs.
GPU-bound stage managed by VRAMJanitor with CPU fallback.
"""

import logging
import numpy as np
from pathlib import Path
from bhashara.models.schema import ProjectMetadata
from bhashara.vram_manager import VRAMJanitor
from bhashara.utils.audio_utils import load_audio, save_audio

logger = logging.getLogger("bhashara.stage2")


def run_stage2_separation(project: ProjectMetadata, project_dir: Path) -> ProjectMetadata:
    """
    Separates input audio into vocals.wav and no_vocals.wav stems.
    Resumable: Skips if vocal_stem_path and instrumental_stem_path already exist.
    """
    stems_dir = project_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = stems_dir / "vocals.wav"
    no_vocal_path = stems_dir / "no_vocals.wav"

    if (
        project.stage_status.get("separation") == "completed"
        and vocal_path.exists()
        and no_vocal_path.exists()
    ):
        logger.info("[Stage 2] Stems already separated. Resuming from cache.")
        project.vocal_stem_path = str(vocal_path)
        project.instrumental_stem_path = str(no_vocal_path)
        return project

    logger.info("[Stage 2] Starting vocal and instrumental stem separation...")
    if not project.audio_16k_mono:
        raise ValueError("Project audio_16k_mono is not set.")
    audio_path = Path(project.audio_16k_mono)
    audio_data, sr = load_audio(audio_path)

    # 1. Manage GPU VRAM Context
    with VRAMJanitor.stage_context("Demucs_Separation", required_vram_gb=1.5) as device:
        separated = False
        try:
            import torch  # type: ignore
            import julius  # type: ignore
            from demucs.apply import apply_model  # type: ignore
            from demucs.pretrained import get_model  # type: ignore
            
            logger.info(f"[Stage 2] Loading Demucs htdemucs model on {device}...")
            model = get_model("htdemucs")
            model.to(device)
            model.eval()
            
            # Format tensor (batch, channels, time) and resample to model samplerate (44.1kHz)
            wav_mono = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            wav_stereo = wav_mono.repeat(1, 2, 1).to(device)
            wav_44k = julius.resample_frac(wav_stereo, sr, model.samplerate)
            
            logger.info(f"[Stage 2] Running Demucs separation on {device}...")
            with torch.no_grad():
                sources = apply_model(model, wav_44k, device=device)
            # Demucs sources: 0: drums, 1: bass, 2: other, 3: vocals
            vocals_44k = sources[0, 3].mean(dim=0, keepdim=True)  # (1, T)
            no_vocals_44k = sources[0, :3].sum(dim=0).mean(dim=0, keepdim=True)  # (1, T)
            
            # Resample back to original sr (16kHz)
            vocals_16k = julius.resample_frac(vocals_44k, model.samplerate, sr).squeeze(0).cpu().numpy()
            no_vocals_16k = julius.resample_frac(no_vocals_44k, model.samplerate, sr).squeeze(0).cpu().numpy()
            
            # Align lengths
            min_len_v = min(len(audio_data), len(vocals_16k))
            min_len_nv = min(len(audio_data), len(no_vocals_16k))
            v_out = np.zeros_like(audio_data)
            nv_out = np.zeros_like(audio_data)
            v_out[:min_len_v] = vocals_16k[:min_len_v]
            nv_out[:min_len_nv] = no_vocals_16k[:min_len_nv]
            
            save_audio(vocal_path, v_out, sr)
            save_audio(no_vocal_path, nv_out, sr)
            separated = True
            logger.info("[Stage 2] Demucs separation completed successfully.")
            del model, wav_mono, wav_stereo, wav_44k, sources, vocals_44k, no_vocals_44k
        except Exception as e:
            logger.warning(f"[Stage 2] Native Demucs separation failed ({e}). Using spectral harmonic separation fallback.")

        if not separated:
            # Fallback: Spectral harmonic separation
            fft_data = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), d=1.0 / sr)
            
            vocal_mask = (freqs >= 80) & (freqs <= 3800)
            vocal_fft = fft_data * vocal_mask
            no_vocal_fft = fft_data * (~vocal_mask)
            
            vocals_np = np.fft.irfft(vocal_fft, n=len(audio_data)).astype(np.float32)
            no_vocals_np = np.fft.irfft(no_vocal_fft, n=len(audio_data)).astype(np.float32)
            
            save_audio(vocal_path, vocals_np, sr)
            save_audio(no_vocal_path, no_vocals_np * 0.8, sr)

    project.vocal_stem_path = str(vocal_path)
    project.instrumental_stem_path = str(no_vocal_path)
    project.stage_status["separation"] = "completed"
    project.save(project_dir / "segments.json")
    
    logger.info("[Stage 2] Stem separation complete: vocals and no_vocals saved.")
    return project
