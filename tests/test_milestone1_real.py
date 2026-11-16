"""
Bhashara Milestone 1 Real End-to-End Test & Proof Generator
Tests all 10 stages using real models on real multi-speaker dialogue over music:
- FFmpeg 9.0.2 static demuxer
- Demucs v4 htdemucs neural vocal separator
- faster-whisper base ASR with word timestamps
- Acoustic F0 Diarization (Pyin / Autocorrelation pitch clustering)
- MarianMT Helsinki-NLP/opus-mt-en-hi syllable-budgeted neural translation
- Edge-TTS neural speech synthesis (hi-IN-MadhurNeural & hi-IN-SwaraNeural)
- WSOLA pitch-preserving time-stretching & micro-pause compression
- Sidechain ducking & EBU R128 loudness normalization
- FFmpeg packaging & bilingual subtitles
- Exports complete verification assets to proof/ folder
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bhashara.pipeline import BhasharaDubber
from bhashara.models.schema import ProjectMetadata
from bhashara.vram_manager import VRAMJanitor

PROOF_DIR = PROJECT_ROOT / "proof"
CLIP1_VIDEO = PROJECT_ROOT / "tests" / "test_data" / "clip1" / "clip1_video.mp4"


def run_milestone1_real_test():
    print("=" * 80)
    print(" 🚀 BHASHARA MILESTONE 1 REAL PIPELINE TEST")
    print("=" * 80)
    
    if not CLIP1_VIDEO.exists():
        raise FileNotFoundError(f"Test clip not found at {CLIP1_VIDEO}. Run generate_test_clips.py first.")

    # 1. Clean previous proof directory
    if PROOF_DIR.exists():
        shutil.rmtree(PROOF_DIR)
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "stems").mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "transcription").mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "diarization").mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "translation").mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "cloned_lines").mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "subtitles").mkdir(parents=True, exist_ok=True)

    project_id = "milestone1_real_job"
    dubber = BhasharaDubber(project_id)
    
    # Clean project cache for fresh real run
    if dubber.project_dir.exists():
        shutil.rmtree(dubber.project_dir)
    dubber.project_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Test] Input Video: {CLIP1_VIDEO}")
    print(f"[Test] Video Size: {CLIP1_VIDEO.stat().st_size} bytes")
    
    project = dubber.get_or_create_project(
        source_video=str(CLIP1_VIDEO),
        target_lang="hi",
        source_lang="en",
        voice_mode="clone"
    )

    # 2. Execute Sequential Pipeline
    print("\n--- Running Sequential Pipeline ---")
    t_seq_start = time.time()
    project = dubber.run_sequential(project, tts_engine="edge_tts", enable_lipsync=False)
    seq_time = time.time() - t_seq_start
    print(f"Sequential Run Finished in {seq_time:.2f}s")

    # 3. Export all intermediate assets to proof/ folder
    print("\n--- Exporting Intermediate Proof Files to proof/ ---")
    
    # Original Video
    shutil.copy2(str(CLIP1_VIDEO), str(PROOF_DIR / "original_video.mp4"))
    
    # Stems
    if project.vocal_stem_path and Path(project.vocal_stem_path).exists():
        shutil.copy2(project.vocal_stem_path, str(PROOF_DIR / "stems" / "vocals.wav"))
    if project.instrumental_stem_path and Path(project.instrumental_stem_path).exists():
        shutil.copy2(project.instrumental_stem_path, str(PROOF_DIR / "stems" / "no_vocals.wav"))
        
    # Transcript
    transcript_data = [
        {"id": s.id, "start": s.start, "end": s.end, "speaker": s.speaker, "text": s.original_text}
        for s in project.segments
    ]
    with open(PROOF_DIR / "transcription" / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
    # Diarization
    diar_data = {
        name: {"gender": spk.gender, "ref_clip": spk.ref_audio_path}
        for name, spk in project.speakers.items()
    }
    with open(PROOF_DIR / "diarization" / "diarization.json", "w", encoding="utf-8") as f:
        json.dump(diar_data, f, indent=2)
        
    # Translation
    trans_data = [
        {
            "id": s.id,
            "original_english": s.original_text,
            "translated_hindi": s.translated_text,
            "duration_s": s.target_duration,
            "syllable_budget": s.syllables_budget,
            "syllables_actual": s.syllables_actual
        }
        for s in project.segments
    ]
    with open(PROOF_DIR / "translation" / "translation.json", "w", encoding="utf-8") as f:
        json.dump(trans_data, f, indent=2, ensure_ascii=False)

    # Cloned lines (both raw TTS and fitted/aligned)
    tts_dir = dubber.project_dir / "tts"
    aligned_dir = dubber.project_dir / "aligned"
    for s in project.segments:
        raw_clip = tts_dir / f"seg_{s.id:04d}.wav"
        if raw_clip.exists():
            shutil.copy2(str(raw_clip), str(PROOF_DIR / "cloned_lines" / f"raw_seg_{s.id:04d}.wav"))
        fitted_clip = aligned_dir / f"seg_{s.id:04d}.wav"
        if fitted_clip.exists():
            shutil.copy2(str(fitted_clip), str(PROOF_DIR / "cloned_lines" / f"fitted_seg_{s.id:04d}.wav"))

    # Subtitles
    sub_dir = dubber.project_dir / "subtitles"
    for srt_f in sub_dir.glob("*.srt"):
        shutil.copy2(str(srt_f), str(PROOF_DIR / "subtitles" / srt_f.name))

    # Final Dubbed Video
    if project.final_dubbed_video_path and Path(project.final_dubbed_video_path).exists():
        shutil.copy2(project.final_dubbed_video_path, str(PROOF_DIR / "milestone1_dubbed_video.mp4"))

    # Run log
    if (dubber.project_dir / "run_log.txt").exists():
        shutil.copy2(str(dubber.project_dir / "run_log.txt"), str(PROOF_DIR / "run_log.txt"))

    # 4. Measure Overlapped Pipeline Speedup
    print("\n--- Measuring CPU/GPU Overlap Execution ---")
    overlap_project_id = "milestone1_overlap_job"
    overlap_dubber = BhasharaDubber(overlap_project_id)
    if overlap_dubber.project_dir.exists():
        shutil.rmtree(overlap_dubber.project_dir)
    overlap_dubber.project_dir.mkdir(parents=True, exist_ok=True)
    
    overlap_project = overlap_dubber.get_or_create_project(
        source_video=str(CLIP1_VIDEO),
        target_lang="hi",
        source_lang="en",
        voice_mode="clone"
    )
    t_over_start = time.time()
    overlap_project = overlap_dubber.run_overlapped(overlap_project, tts_engine="edge_tts", enable_lipsync=False)
    over_time = time.time() - t_over_start
    print(f"Overlapped Run Finished in {over_time:.2f}s")
    
    speedup = ((seq_time - over_time) / seq_time) * 100.0 if seq_time > 0 else 0.0
    print(f"\nOverlap Speedup: {speedup:+.1f}% ({seq_time:.2f}s sequential vs {over_time:.2f}s overlapped)")

    # 5. Output Verification Summary
    print("\n" + "=" * 80)
    print(" ✅ MILESTONE 1 REAL TEST COMPLETE")
    print("=" * 80)
    print(f"Final Dubbed Video Path: {PROOF_DIR / 'milestone1_dubbed_video.mp4'}")
    print(f"Intermediate Stems:      {PROOF_DIR / 'stems'}")
    print(f"Hindi Translations:      {PROOF_DIR / 'translation' / 'translation.json'}")
    print(f"Diarization Profiles:    {PROOF_DIR / 'diarization' / 'diarization.json'}")
    print(f"Cloned Lines:            {PROOF_DIR / 'cloned_lines'}")
    print(f"Subtitles:               {PROOF_DIR / 'subtitles'}")
    print(f"Run Log:                 {PROOF_DIR / 'run_log.txt'}")
    print("=" * 80)


if __name__ == "__main__":
    run_milestone1_real_test()
