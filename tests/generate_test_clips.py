"""
Generates real multi-speaker dialogue test clips with real speech and background music.
Clip 1: 14s conversation between Male (Speaker 1) and Female (Speaker 2) over music.
Clip 2: 12s completely different conversation between Female (Speaker 1) and Male (Speaker 2) over music.
"""

import asyncio
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import edge_tts
from bhashara.utils.audio_utils import get_ffmpeg_path
TESTS_DIR = PROJECT_ROOT / "tests" / "test_data"
TESTS_DIR.mkdir(parents=True, exist_ok=True)


async def synthesize_speech(text: str, voice: str, out_wav: Path):
    """Generates natural neural speech via Edge-TTS and converts to 16kHz WAV."""
    temp_mp3 = out_wav.with_suffix(".mp3")
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(temp_mp3))
    
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-i", str(temp_mp3),
        "-ar", "16000", "-ac", "1", str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if temp_mp3.exists():
        temp_mp3.unlink()


def create_ambient_music(duration_sec: float, sr: int = 16000, key_freq: float = 220.0) -> np.ndarray:
    """Synthesizes an ambient harmonic background music track with chord progression."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Chord progression: Am -> F -> C -> G
    chord_len = duration_sec / 4.0
    music = np.zeros_like(t)
    
    chords = [
        [key_freq, key_freq * 1.2, key_freq * 1.5],          # Minor
        [key_freq * 0.89, key_freq * 1.12, key_freq * 1.33], # Major
        [key_freq * 1.33, key_freq * 1.67, key_freq * 2.0],  # Major
        [key_freq * 1.5, key_freq * 1.88, key_freq * 2.25]   # Dominant
    ]
    
    for i, chord in enumerate(chords):
        idx_start = int(i * chord_len * sr)
        idx_end = int((i + 1) * chord_len * sr)
        t_sub = t[idx_start:idx_end]
        sub_music = np.zeros_like(t_sub)
        for freq in chord:
            sub_music += 0.05 * np.sin(2 * np.pi * freq * t_sub)
            sub_music += 0.02 * np.sin(2 * np.pi * freq * 2.0 * t_sub) # harmonic
        # Envelope: subtle fade
        music[idx_start:idx_end] = sub_music
        
    # Soft low-pass smoothing & scale volume
    return (music * 0.15).astype(np.float32)


async def generate_clip1():
    """Generates Clip 1: 14 seconds dual-speaker conversation."""
    clip1_dir = TESTS_DIR / "clip1"
    clip1_dir.mkdir(parents=True, exist_ok=True)
    
    male_wav = clip1_dir / "spk1_male.wav"
    fem_wav = clip1_dir / "spk2_female.wav"
    
    text1 = "Welcome to the studio. We are testing the video dubbing pipeline today."
    text2 = "Yes, I am excited to see how our voices sound in Hindi."
    
    print("Generating speech for Clip 1...")
    await synthesize_speech(text1, "en-US-GuyNeural", male_wav)
    await synthesize_speech(text2, "en-US-JennyNeural", fem_wav)
    
    d1, sr = sf.read(str(male_wav))
    d2, _ = sf.read(str(fem_wav))
    
    total_sec = 14.0
    master_audio = np.zeros(int(total_sec * sr), dtype=np.float32)
    
    # Place Speaker 1 at t=0.6s
    pos1 = int(0.6 * sr)
    master_audio[pos1:pos1 + len(d1)] += d1
    
    # Place Speaker 2 at t=7.0s
    pos2 = int(7.0 * sr)
    master_audio[pos2:pos2 + len(d2)] += d2
    
    # Add ambient background music
    bgm = create_ambient_music(total_sec, sr, key_freq=220.0)
    master_audio += bgm
    
    # Peak normalize
    master_audio = master_audio / (np.max(np.abs(master_audio)) + 1e-6) * 0.95
    
    mixed_wav = clip1_dir / "clip1_audio.wav"
    sf.write(str(mixed_wav), master_audio, sr)
    
    # Generate MP4 video with FFmpeg testsrc
    video_mp4 = clip1_dir / "clip1_video.mp4"
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={total_sec}:size=1280x720:rate=24",
        "-i", str(mixed_wav),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        str(video_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"Clip 1 generated: {video_mp4} ({video_mp4.stat().st_size} bytes)")
    return video_mp4


async def generate_clip2():
    """Generates Clip 2: 12 seconds distinct conversation."""
    clip2_dir = TESTS_DIR / "clip2"
    clip2_dir.mkdir(parents=True, exist_ok=True)
    
    fem_wav = clip2_dir / "spk1_female.wav"
    male_wav = clip2_dir / "spk2_male.wav"
    
    text1 = "Did you check the audio levels on the microphone?"
    text2 = "Everything is calibrated and ready for the recording session."
    
    print("Generating speech for Clip 2...")
    await synthesize_speech(text1, "en-US-JennyNeural", fem_wav)
    await synthesize_speech(text2, "en-US-GuyNeural", male_wav)
    
    d1, sr = sf.read(str(fem_wav))
    d2, _ = sf.read(str(male_wav))
    
    total_sec = 12.0
    master_audio = np.zeros(int(total_sec * sr), dtype=np.float32)
    
    # Place Speaker 1 at t=0.5s
    pos1 = int(0.5 * sr)
    master_audio[pos1:pos1 + len(d1)] += d1
    
    # Place Speaker 2 at t=5.5s
    pos2 = int(5.5 * sr)
    master_audio[pos2:pos2 + len(d2)] += d2
    
    # Add ambient background music (different key)
    bgm = create_ambient_music(total_sec, sr, key_freq=174.6) # F note
    master_audio += bgm
    
    master_audio = master_audio / (np.max(np.abs(master_audio)) + 1e-6) * 0.95
    
    mixed_wav = clip2_dir / "clip2_audio.wav"
    sf.write(str(mixed_wav), master_audio, sr)
    
    video_mp4 = clip2_dir / "clip2_video.mp4"
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={total_sec}:size=1280x720:rate=24",
        "-i", str(mixed_wav),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        str(video_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"Clip 2 generated: {video_mp4} ({video_mp4.stat().st_size} bytes)")
    return video_mp4


async def main():
    await generate_clip1()
    await generate_clip2()
    print("All test clips generated successfully.")


if __name__ == "__main__":
    asyncio.run(main())
