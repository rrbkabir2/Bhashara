"""
Bhashara Local Web Server (ui/server.py)
FastAPI application bound strictly to localhost (127.0.0.1) providing:
- Media upload and URL pre-flight info extraction
- Real-time job status and stage progress streaming
- Full audio/video streaming with HTTP range headers
- Interactive script editor API with single-line re-dubbing
- Media download endpoints for video, audio, stems, and subtitles
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ui.job_manager import JobManager, UPLOADS_DIR, OUTPUT_DIR
from bhashara.utils.ytdlp_utils import get_video_info
from bhashara.utils.audio_utils import get_video_duration_sec

logger = logging.getLogger("bhashara.server")

app = FastAPI(title="Bhashara Local Video Dubbing Studio")
job_manager = JobManager()

STATIC_DIR = PROJECT_ROOT / "ui" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class URLInfoRequest(BaseModel):
    url: str


class CreateURLJobRequest(BaseModel):
    url: str
    target_lang: str = "hi"
    source_lang: str = "en"
    title: Optional[str] = None
    duration: Optional[float] = 0.0


class RedubLineRequest(BaseModel):
    segment_id: int
    new_text: str
    new_speaker: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main single-page web studio interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Bhashara Studio UI initializing... please refresh.</h1>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/info")
async def query_video_info(req: URLInfoRequest):
    """
    Pre-flight query for YouTube / web links.
    Returns title, duration, thumbnail, and plain-English error if inaccessible.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please enter a valid video link.")

    info = get_video_info(url)
    if not info.get("success"):
        return JSONResponse(status_code=400, content={"error": info.get("error")})

    dur = info.get("duration", 0.0)
    # Estimate time on CPU (typically ~2.5x to 3.0x video duration on CPU)
    estimated_sec = dur * 2.8
    est_mins = max(1, int(estimated_sec // 60))

    return {
        "success": True,
        "title": info.get("title"),
        "duration": dur,
        "duration_formatted": info.get("duration_formatted"),
        "thumbnail": info.get("thumbnail"),
        "estimated_dub_mins": est_mins,
        "needs_length_warning": dur > 300.0,  # Warning if longer than 5 minutes
        "warning_message": (
            f"This video is {info.get('duration_formatted')} long. Because Bhashara is running on your CPU, "
            f"dubbing will take approximately {est_mins} to {est_mins + 2} minutes. "
            "Do you want to proceed?"
        ) if dur > 300.0 else None
    }


@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    target_lang: str = Form("hi"),
    source_lang: str = Form("en"),
    title: Optional[str] = Form(None),
    duration: Optional[float] = Form(0.0)
):
    """Creates and queues a new dubbing job from either an uploaded file or a URL."""
    if file and file.filename:
        # Save uploaded file
        safe_name = Path(file.filename).name
        dest_path = UPLOADS_DIR / f"{int(time_now())}_{safe_name}"
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect duration
        file_dur = get_video_duration_sec(dest_path)
        job_id = job_manager.create_job(
            source_path_or_url=str(dest_path),
            is_url=False,
            target_lang=target_lang,
            source_lang=source_lang,
            title=safe_name,
            duration=file_dur
        )
        return {"success": True, "job_id": job_id, "status": "queued"}

    elif url and url.strip():
        # URL ingestion
        job_id = job_manager.create_job(
            source_path_or_url=url.strip(),
            is_url=True,
            target_lang=target_lang,
            source_lang=source_lang,
            title=title or "Online Video",
            duration=duration or 0.0
        )
        return {"success": True, "job_id": job_id, "status": "queued"}

    else:
        raise HTTPException(status_code=400, detail="Please upload a video file or provide a video URL.")


def time_now():
    import time
    return time.time()


@app.get("/api/jobs")
async def list_jobs():
    """Returns list of previous jobs and disk space usage."""
    jobs = job_manager.list_jobs()
    disk = job_manager.get_disk_usage()
    return {
        "jobs": jobs,
        "disk": disk,
        "active_job_id": job_manager.active_job_id
    }


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Returns real-time progress, stages, logs, and outputs for a specific job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancels a job and cleans up hardware memory."""
    ok = job_manager.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not cancel job.")
    return {"success": True, "message": "Job cancelled successfully."}


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resumes an interrupted or cancelled job from cached stage checkpoints."""
    ok = job_manager.resume_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not resume job.")
    return {"success": True, "message": "Job resumed."}


@app.post("/api/jobs/{job_id}/redub")
async def redub_line(job_id: str, req: RedubLineRequest):
    """Granularly re-dubs a single line in a completed job."""
    try:
        res = job_manager.redub_line(
            job_id,
            segment_id=req.segment_id,
            new_text=req.new_text,
            new_speaker=req.new_speaker
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Re-dubbing failed: {e}")


@app.post("/api/clean_temp")
async def clean_temporary_files():
    """Deletes intermediate chunk files while preserving finished outputs."""
    res = job_manager.clean_temp_files()
    disk = job_manager.get_disk_usage()
    return {"success": True, "cleaned_mb": res["cleaned_mb"], "disk": disk}


@app.get("/media/{file_path:path}")
async def stream_media(file_path: str):
    """Streams local audio/video with HTTP Range support for browser players."""
    resolved = Path(file_path)
    if not resolved.exists():
        # Fallback relative to project root
        resolved = PROJECT_ROOT / file_path

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")

    media_type = "video/mp4"
    if resolved.suffix.lower() == ".wav":
        media_type = "audio/wav"
    elif resolved.suffix.lower() == ".mp3":
        media_type = "audio/mpeg"

    return FileResponse(resolved, media_type=media_type)


@app.get("/download/{job_id}/{file_type}")
async def download_output_asset(job_id: str, file_type: str):
    """Handles direct file downloads for video, audio, stems, and subtitles."""
    job = job_manager.get_job(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job outputs not ready.")

    outputs = job.get("outputs", {})
    target_path = None
    filename = None

    if file_type == "video":
        target_path = outputs.get("dubbed_video")
        filename = f"{job['title']}_dubbed.mp4"
    elif file_type == "audio":
        target_path = outputs.get("dubbed_audio")
        filename = f"{job['title']}_dubbed_audio.wav"
    elif file_type == "vocals":
        target_path = outputs.get("vocals_stem")
        filename = f"{job['title']}_vocals_isolated.wav"
    elif file_type == "music":
        target_path = outputs.get("music_stem")
        filename = f"{job['title']}_music_effects.wav"
    elif file_type in ("original_sub", "translated_sub", "bilingual_sub"):
        sub_key = file_type.split("_")[0]
        target_path = outputs.get("subtitles", {}).get(sub_key)
        filename = f"{job['title']}_{file_type}.srt"

    if not target_path or not Path(target_path).exists():
        raise HTTPException(status_code=404, detail=f"File {file_type} not found.")

    return FileResponse(target_path, filename=filename, media_type="application/octet-stream")
