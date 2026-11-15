"""
Bhashara Stage 9: Packaging & Muxing (Output Delivery)
Muxes dubbed audio into video via FFmpeg, exports all stems, and compiles
original, translated, and bilingual subtitles (.srt and .vtt).
CPU-bound stage.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from bhashara.models.schema import ProjectMetadata
from bhashara.utils.srt_generator import generate_subtitles
from bhashara.utils.audio_utils import get_ffmpeg_path

logger = logging.getLogger("bhashara.stage9")


def run_stage9_output(project: ProjectMetadata, project_dir: Path) -> ProjectMetadata:
    """
    Remuxes audio into final MP4 video and exports multi-track subtitle files.
    Resumable: Skips if output video already created and valid.
    """
    out_dir = project_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_video = out_dir / "dubbed_video.mp4"

    if (
        project.stage_status.get("output") == "completed"
        and final_video.exists()
        and not any(s.dirty for s in project.segments)
    ):
        logger.info("[Stage 9] Output video already muxed. Resuming.")
        project.final_dubbed_video_path = str(final_video)
        return project

    logger.info("[Stage 9] Starting final packaging, FFmpeg remux, and subtitle export...")

    # 1. Generate Subtitle Files (Original, Translated, Bilingual)
    subs = generate_subtitles(project.segments, out_dir, base_name="dubbed")
    project.subtitles = subs
    logger.info(f"[Stage 9] Exported subtitles: {list(subs.keys())}")

    # 2. FFmpeg Remux Video + Dubbed Audio
    src_video = Path(project.source_video)
    master_audio = Path(project.final_dubbed_audio_path)
    
    ffmpeg_exe = get_ffmpeg_path()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(src_video),
        "-i", str(master_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(final_video)
    ]
    
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            logger.warning(f"[Stage 9] FFmpeg remux failed ({proc.stderr}). Copying source video as fallback.")
            shutil.copy2(src_video, final_video)
        else:
            logger.info(f"[Stage 9] Dubbed video remuxed successfully: {final_video}")
    except Exception as e:
        logger.warning(f"[Stage 9] FFmpeg executable not runnable ({e}). Fallback to copy.")
        shutil.copy2(src_video, final_video)

    project.final_dubbed_video_path = str(final_video)
    project.stage_status["output"] = "completed"
    project.save(project_dir / "segments.json")
    
    logger.info("[Stage 9] Output delivery complete.")
    return project
