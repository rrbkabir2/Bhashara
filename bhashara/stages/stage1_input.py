"""
Bhashara Stage 1: Input Ingestion & Audio Demuxing
Accepts local video or URL (YouTube), extracts 16kHz mono PCM WAV via FFmpeg.
CPU-bound stage.
"""

import shutil
import subprocess
import logging
from pathlib import Path
from bhashara.models.schema import ProjectMetadata
from bhashara.utils.audio_utils import extract_audio_16k_mono

logger = logging.getLogger("bhashara.stage1")


def run_stage1_input(project: ProjectMetadata, project_dir: Path) -> ProjectMetadata:
    """
    Ingests source video and produces 16kHz mono audio.
    Resumable: Skips if audio_16k_mono already exists and stage is completed.
    """
    if project.stage_status.get("input") == "completed" and project.audio_16k_mono:
        if Path(project.audio_16k_mono).exists():
            logger.info("[Stage 1] Input already processed. Resuming from cache.")
            return project

    logger.info(f"[Stage 1] Ingesting input: {project.source_video}")
    project_dir.mkdir(parents=True, exist_ok=True)
    
    src = project.source_video
    local_video_path = project_dir / "source_video.mp4"
    audio_wav_path = project_dir / "source_audio_16k.wav"

    # 1. URL Ingestion (YouTube / direct link)
    if src.startswith("http://") or src.startswith("https://"):
        logger.info(f"[Stage 1] Downloading link via yt-dlp: {src}")
        try:
            from bhashara.utils.ytdlp_utils import download_video
            download_video(src, local_video_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download video link with yt-dlp: {e}")
    else:
        # Local file
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Source video file not found: {src}")
        if src_path != local_video_path:
            shutil.copy2(src_path, local_video_path)

    project.source_video = str(local_video_path)

    # 2. Extract 16kHz Mono WAV via FFmpeg
    logger.info(f"[Stage 1] Extracting 16kHz mono audio -> {audio_wav_path}")
    try:
        extract_audio_16k_mono(local_video_path, audio_wav_path)
    except Exception as e:
        logger.warning(f"[Stage 1] FFmpeg extraction failed or ffmpeg missing ({e}). Checking if source is WAV...")
        if Path(src).suffix.lower() == ".wav":
            shutil.copy2(src, audio_wav_path)
        else:
            raise

    project.audio_16k_mono = str(audio_wav_path)
    project.stage_status["input"] = "completed"
    project.save(project_dir / "segments.json")
    
    logger.info("[Stage 1] Input stage completed successfully.")
    return project
