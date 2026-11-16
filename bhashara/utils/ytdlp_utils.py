"""
Bhashara yt-dlp Video Ingestion Utilities
Extracts metadata (title, duration, thumbnail) and downloads online video streams
with user-friendly plain-English error reporting for private, age-restricted, or live videos.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("bhashara.ytdlp")


def get_video_info(url: str) -> Dict[str, Any]:
    """
    Fetches video metadata (title, duration, thumbnail, live status)
    without downloading the media. Returns plain-English error on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        return {
            "success": False,
            "error": "yt-dlp library is not installed. Please run: pip install yt-dlp"
        }

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"success": False, "error": "Could not retrieve video information."}

            # Check if live stream
            is_live = info.get("is_live", False) or info.get("was_live", False) and not info.get("duration")
            if is_live:
                return {
                    "success": False,
                    "error": "This video is a live stream. Bhashara only dubs completed, recorded videos."
                }

            duration = float(info.get("duration", 0) or 0)
            title = info.get("title", "Online Video")
            thumbnail = info.get("thumbnail") or (info.get("thumbnails", [{}])[-1].get("url") if info.get("thumbnails") else "")

            return {
                "success": True,
                "title": title,
                "duration": duration,
                "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
                "thumbnail": thumbnail,
                "uploader": info.get("uploader", "Unknown"),
                "view_count": info.get("view_count", 0),
                "error": None
            }

    except Exception as e:
        err_msg = str(e)
        logger.warning(f"yt-dlp extract_info failed for '{url}': {err_msg}")
        
        # User-friendly plain English errors
        if "Private video" in err_msg:
            friendly = "This video is private. Please provide a public or unlisted link."
        elif "Sign in to confirm your age" in err_msg or "age-gated" in err_msg.lower():
            friendly = "This video is age-restricted and cannot be downloaded without login."
        elif "not available in your country" in err_msg or "geo-restricted" in err_msg.lower():
            friendly = "This video is blocked in your region."
        elif "Premieres in" in err_msg or "upcoming" in err_msg.lower():
            friendly = "This video has not premiered yet."
        elif "Incomplete YouTube ID" in err_msg or "not a valid URL" in err_msg.lower():
            friendly = "The link is invalid or incomplete. Please check the URL and try again."
        elif "Video unavailable" in err_msg:
            friendly = "This video is unavailable or has been removed."
        else:
            friendly = f"Could not open this video link: {err_msg.split('ERROR:')[-1].strip()}"

        return {
            "success": False,
            "error": friendly
        }


def download_video(url: str, output_path: Path, progress_callback=None) -> Path:
    """
    Downloads online video to output_path using best mp4 format compatible with FFmpeg.
    """
    import yt_dlp

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_template = str(output_path.with_suffix("")) + ".%(ext)s"

    def _hook(d):
        if progress_callback and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
            downloaded = d.get("downloaded_bytes", 0)
            pct = min(100.0, max(0.0, (downloaded / total) * 100.0))
            progress_callback(pct, d.get("_speed_str", ""))

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": temp_template,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook] if progress_callback else [],
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        err_msg = str(e)
        if "Private video" in err_msg:
            raise RuntimeError("This video is private. Please use a public link.")
        elif "Sign in to confirm your age" in err_msg:
            raise RuntimeError("This video is age-restricted.")
        elif "not available in your country" in err_msg:
            raise RuntimeError("This video is region-locked.")
        else:
            raise RuntimeError(f"Video download failed: {err_msg.split('ERROR:')[-1].strip()}")

    # Find resulting file
    candidates = list(output_path.parent.glob(output_path.stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    # If extension differs or merged to mp4
    mp4_candidates = [c for c in candidates if c.suffix.lower() == ".mp4"]
    actual_file = mp4_candidates[0] if mp4_candidates else candidates[0]
    
    if actual_file != output_path:
        if output_path.exists():
            output_path.unlink()
        actual_file.rename(output_path)

    return output_path
