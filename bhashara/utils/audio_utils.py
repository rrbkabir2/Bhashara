"""
Bhashara Audio Utilities
Professional DSP algorithms: pitch F0 estimation, SOLA time-stretch, sidechain ducking,
EBU R128 loudness normalization, and robust FFmpeg wrappers.
"""

import os
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import wave

# Audio target constants
TARGET_LUFS = -16.0
TRUE_PEAK_LIMIT = 0.891  # -1.0 dBTP in linear amplitude


def get_ffmpeg_path() -> str:
    """Finds bundled tools/ffmpeg.exe, system ffmpeg, or imageio_ffmpeg binary."""
    # 0. Check bundled tools/ffmpeg.exe
    project_root = Path(__file__).resolve().parent.parent.parent
    bundled = project_root / "tools" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)

    # 1. System PATH
    ffmpeg_sys = shutil.which("ffmpeg")
    if ffmpeg_sys:
        return ffmpeg_sys
        
    # 2. Check imageio_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
        
    # 3. Standard paths on Windows
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe"
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
            
    return "ffmpeg"


def extract_audio_16k_mono(video_path: Path, output_wav: Path) -> Path:
    """Extracts 16kHz 16-bit mono PCM WAV for ASR and Diarization via FFmpeg."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav)
    ]
    
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {proc.stderr}")
    return output_wav


def load_audio(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Loads audio file into float32 numpy array normalized to [-1.0, 1.0]."""
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        # Standard library wave fallback
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            raw = wf.readframes(n_frames)
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            return data, sr


def save_audio(wav_path: Path, data: np.ndarray, sr: int) -> None:
    """Saves float32 numpy array as 16-bit PCM WAV."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    # Clip to prevent distortion
    clipped = np.clip(data, -1.0, 1.0)
    try:
        import soundfile as sf  # type: ignore
        sf.write(str(wav_path), clipped, sr, subtype="PCM_16")
    except Exception:
        # Standard library wave fallback
        int_data = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_data.tobytes())


def estimate_pitch_f0(audio: np.ndarray, sr: int) -> Optional[float]:
    """
    Computes median pitch (F0) using normalized autocorrelation.
    Gender threshold: Median F0 > 165 Hz -> female, <= 165 Hz -> male.
    """
    if len(audio) < sr * 0.2:
        return None
        
    frame_len = int(sr * 0.05)  # 50ms window
    hop_len = int(sr * 0.025)
    f0_values = []
    
    # Target voice range: 60 Hz to 400 Hz
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)
    
    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i:i + frame_len]
        # Energy gate
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.02:
            continue
            
        frame = frame - np.mean(frame)
        autocorr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) == 0:
            continue
            
        peak_idx = np.argmax(search_region) + min_lag
        peak_val = autocorr[peak_idx]
        
        if peak_val > 0.3 * autocorr[0]:
            f0 = sr / peak_idx
            f0_values.append(f0)
            
    if not f0_values:
        return 130.0  # default neutral
    return float(np.median(f0_values))


def time_stretch_sola(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """
    Synchronous Overlap-Add (SOLA) pitch-preserving time stretch.
    Expands or compresses time without changing vocal pitch.
    """
    if abs(speed - 1.0) < 0.02:
        return audio
        
    speed = np.clip(speed, 0.75, 1.30)
    win_size = int(sr * 0.04)   # 40ms frame
    hop_in = int(win_size / 2)
    hop_out = int(hop_in / speed)
    max_delta = int(win_size / 4)
    
    window = np.hanning(win_size)
    num_frames = (len(audio) - win_size) // hop_in
    if num_frames <= 1:
        return audio
        
    out_len = int(len(audio) / speed) + win_size
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    
    pos_out = 0
    for f in range(num_frames):
        pos_in = f * hop_in
        frame = audio[pos_in:pos_in + win_size] * window
        
        # Best alignment in overlapping region
        best_delta = 0
        if f > 0:
            overlap_region = output[pos_out:pos_out + win_size]
            best_corr = -1e9
            for d in range(-max_delta, max_delta, 2):
                if pos_out + d < 0 or pos_out + d + win_size > out_len:
                    continue
                corr = np.dot(frame, output[pos_out + d:pos_out + d + win_size])
                if corr > best_corr:
                    best_corr = corr
                    best_delta = d
                    
        target_pos = max(0, pos_out + best_delta)
        if target_pos + win_size <= out_len:
            output[target_pos:target_pos + win_size] += frame
            norm[target_pos:target_pos + win_size] += window
            
        pos_out += hop_out
        
    mask = norm > 1e-3
    output[mask] /= norm[mask]
    return output[:int(len(audio) / speed)]


def compress_pauses(audio: np.ndarray, sr: int, max_silence_sec: float = 0.10) -> np.ndarray:
    """
    Finds internal pauses in dialogue longer than max_silence_sec and truncates them.
    Saves critical time budget without altering spoken phonemes.
    """
    frame_len = int(sr * 0.02)
    hop = frame_len // 2
    max_silence_frames = int(max_silence_sec * sr / hop)
    
    energy = np.array([
        np.sum(audio[i:i + frame_len] ** 2)
        for i in range(0, len(audio) - frame_len, hop)
    ])
    
    silence_threshold = np.mean(energy) * 0.05
    is_silent = energy < silence_threshold
    
    keep_indices = []
    consecutive_silent = 0
    
    for idx, silent in enumerate(is_silent):
        if silent:
            consecutive_silent += 1
            if consecutive_silent <= max_silence_frames:
                keep_indices.append(idx)
        else:
            consecutive_silent = 0
            keep_indices.append(idx)
            
    out_samples = []
    for idx in keep_indices:
        start = idx * hop
        out_samples.append(audio[start:start + hop])
        
    if not out_samples:
        return audio
    return np.concatenate(out_samples)


def normalize_loudness_lufs(audio: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """
    Normalizes audio loudness to target LUFS using RMS integration and true-peak limiting.
    """
    rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
    if rms < 1e-4:
        return audio
        
    # Standard approx: 0 dBFS RMS ~= -3.01 LUFS for speech
    current_lufs = 20 * np.log10(rms) - 3.0
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20.0)
    
    normalized = audio * gain_linear
    peak = np.max(np.abs(normalized))
    if peak > TRUE_PEAK_LIMIT:
        normalized = (normalized / peak) * TRUE_PEAK_LIMIT
        
    return normalized


def sidechain_duck(
    dialogue: np.ndarray,
    bgm: np.ndarray,
    sr: int,
    duck_attenuation_db: float = -10.0,
    attack_ms: int = 50,
    release_ms: int = 250
) -> np.ndarray:
    """
    Dynamic sidechain ducking: Attenuates background music whenever dialogue is active,
    smoothed with gentle attack and release curves.
    """
    max_len = max(len(dialogue), len(bgm))
    d_padded = np.zeros(max_len, dtype=np.float32)
    b_padded = np.zeros(max_len, dtype=np.float32)
    d_padded[:len(dialogue)] = dialogue
    b_padded[:len(bgm)] = bgm
    
    # 1. Dialogue Envelope Follower
    attack_coef = np.exp(-1.0 / (sr * (attack_ms / 1000.0)))
    release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
    
    abs_d = np.abs(d_padded)
    envelope = np.zeros(max_len, dtype=np.float32)
    curr_env = 0.0
    
    for i in range(max_len):
        target = abs_d[i]
        if target > curr_env:
            curr_env = attack_coef * curr_env + (1.0 - attack_coef) * target
        else:
            curr_env = release_coef * curr_env + (1.0 - release_coef) * target
        envelope[i] = curr_env
        
    # 2. Compute Gain Curve
    thresh = 0.02
    min_gain = 10 ** (duck_attenuation_db / 20.0)
    gain_curve = np.ones(max_len, dtype=np.float32)
    
    active_mask = envelope > thresh
    gain_curve[active_mask] = min_gain + (1.0 - min_gain) * (1.0 - np.clip(envelope[active_mask] / 0.2, 0.0, 1.0))
    
    ducked_bgm = b_padded * gain_curve
    return ducked_bgm
