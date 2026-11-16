"""
Bhashara Simple Video Dubbing CLI (dub.py)
Usage:
    python dub.py my_video.mp4
    python dub.py inputs/movie_clip.mp4 --lang hi
"""

import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from bhashara.pipeline import BhasharaDubber


def main():
    parser = argparse.ArgumentParser(description="Bhashara Local Video Dubbing Tool")
    parser.add_argument(
        "video",
        nargs="?",
        default="input.mp4",
        help="Path to the video file to dub (e.g., my_video.mp4 or input.mp4)"
    )
    parser.add_argument(
        "--target-lang",
        "--lang",
        default="hi",
        choices=["hi", "en", "mr"],
        help="Target language: 'hi' for Hindi (default), 'en' for English, 'mr' for Marathi"
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language of the video (default: en)"
    )
    parser.add_argument(
        "--mode",
        default="overlapped",
        choices=["overlapped", "sequential"],
        help="Execution mode: 'overlapped' (faster) or 'sequential'"
    )
    args = parser.parse_args()

    input_path = Path(args.video).resolve()
    if not input_path.exists():
        print("\n❌ ERROR: Video file not found!")
        print(f"Looked for: {input_path}")
        print("\nPlease put your video file in this folder and run:")
        print(f"    python dub.py {Path(args.video).name}\n")
        sys.exit(1)

    print("=" * 80)
    print(" 🎙️  BHASHARA LOCAL VIDEO DUBBING")
    print("=" * 80)
    print(f"Input Video:     {input_path.name}")
    print(f"Target Language: {'Hindi (hi)' if args.target_lang == 'hi' else args.target_lang}")
    print(f"File Size:       {input_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 80)

    # Output directory
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem_name = input_path.stem
    project_id = f"job_{stem_name}"
    dubber = BhasharaDubber(project_id)

    project = dubber.get_or_create_project(
        source_video=str(input_path),
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        voice_mode="clone"
    )

    if args.mode == "overlapped":
        project = dubber.run_overlapped(project, tts_engine="edge_tts", enable_lipsync=False)
    else:
        project = dubber.run_sequential(project, tts_engine="edge_tts", enable_lipsync=False)

    # Save outputs to known, user-friendly paths
    final_video = output_dir / f"dubbed_{stem_name}.mp4"
    if project.final_dubbed_video_path and Path(project.final_dubbed_video_path).exists():
        shutil.copy2(project.final_dubbed_video_path, final_video)

    # Copy proof stems and data
    job_proof_dir = output_dir / f"proof_{stem_name}"
    job_proof_dir.mkdir(parents=True, exist_ok=True)
    
    if project.vocal_stem_path and Path(project.vocal_stem_path).exists():
        shutil.copy2(project.vocal_stem_path, job_proof_dir / "isolated_vocals.wav")
    if project.instrumental_stem_path and Path(project.instrumental_stem_path).exists():
        shutil.copy2(project.instrumental_stem_path, job_proof_dir / "background_music_and_effects.wav")
    if project.final_dubbed_audio_path and Path(project.final_dubbed_audio_path).exists():
        shutil.copy2(project.final_dubbed_audio_path, job_proof_dir / "master_dubbed_audio.wav")
    
    cache_segments = dubber.project_dir / "segments.json"
    if cache_segments.exists():
        shutil.copy2(cache_segments, job_proof_dir / "segments.json")

    sub_dir = dubber.project_dir / "subtitles"
    if sub_dir.exists():
        for srt in sub_dir.glob("*.srt"):
            shutil.copy2(srt, job_proof_dir / srt.name)

    print("\n" + "=" * 80)
    print(" 🎉 DUBBING COMPLETE!")
    print("=" * 80)
    print(f"🎬 Dubbed Video: {final_video}")
    print(f"📁 Proof Files:  {job_proof_dir}")
    print(f"   ├── isolated_vocals.wav               (Demucs separated voice)")
    print(f"   ├── background_music_and_effects.wav  (Kept original music/SFX)")
    print(f"   ├── master_dubbed_audio.wav           (Ducked and mastered audio)")
    print(f"   └── segments.json                     (Transcript & translation)")
    print("=" * 80)


if __name__ == "__main__":
    main()
