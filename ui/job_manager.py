"""
Bhashara Background Job & Concurrency Manager (ui/job_manager.py)
Coordinates single-concurrency video dubbing jobs with real-time stage progress,
true cancellation, resumability, and disk space tracking.
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bhashara.pipeline import BhasharaDubber
from bhashara.vram_manager import VRAMJanitor
from bhashara.utils.audio_utils import get_video_duration_sec

logger = logging.getLogger("bhashara.job_manager")

JOBS_INDEX_FILE = PROJECT_ROOT / "cache" / "job_history.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STAGE_DEFINITIONS = [
    {"num": 1, "name": "Input Demux", "model": "FFmpeg 9.0.2", "desc": "Demuxing media and extracting 16kHz audio"},
    {"num": 2, "name": "Stem Separation", "model": "Demucs v4 (htdemucs)", "desc": "Separating dialogue from background music"},
    {"num": 3, "name": "Transcription", "model": "faster-whisper base", "desc": "Word-level speech recognition and VAD"},
    {"num": 4, "name": "Diarization", "model": "Acoustic F0 Clustering", "desc": "Identifying speakers and pitch profiles"},
    {"num": 5, "name": "Translation", "model": "MarianMT Helsinki-NLP", "desc": "Syllable-budgeted Hindi translation"},
    {"num": 6, "name": "Voice Synthesis", "model": "Edge-TTS Neural", "desc": "Generating natural character voices"},
    {"num": 7, "name": "Timing Fit", "model": "WSOLA SolaStretch", "desc": "Fitting speech timing to original video"},
    {"num": 8, "name": "Ducked Mixer", "model": "Sidechain Ducking -16LUFS", "desc": "Mastering audio with dynamic ducking"},
    {"num": 9, "name": "Video Muxing", "model": "FFmpeg Muxer", "desc": "Remuxing video with new Hindi soundtrack"},
    {"num": 10, "name": "Subtitles & Output", "model": "Bhashara Subtitle Engine", "desc": "Compiling multi-track subtitles"}
]


class JobManager:
    """Singleton Job Manager managing a single worker thread and queued jobs."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self.lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queue: List[str] = []
        self.active_job_id: Optional[str] = None
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._load_job_history()
        self.worker_thread.start()

    def _load_job_history(self):
        """Loads persistent job records from cache/job_history.json."""
        if JOBS_INDEX_FILE.exists():
            try:
                with open(JOBS_INDEX_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    for jid, data in saved.items():
                        if data.get("status") == "running":
                            data["status"] = "failed"
                            data["error"] = "Interrupted by server restart (Resumable)"
                        self.jobs[jid] = data
            except Exception as e:
                logger.warning(f"Could not load job history: {e}")

    def _save_job_history(self):
        """Persists job records to disk."""
        try:
            JOBS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(JOBS_INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(self.jobs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save job history: {e}")

    def create_job(
        self,
        source_path_or_url: str,
        is_url: bool,
        target_lang: str = "hi",
        source_lang: str = "en",
        title: Optional[str] = None,
        duration: float = 0.0
    ) -> str:
        """Enqueues a new dubbing job."""
        timestamp = int(time.time())
        clean_title = (title or (Path(source_path_or_url).stem if not is_url else "web_video"))[:30]
        safe_slug = "".join(c if c.isalnum() else "_" for c in clean_title).strip("_")
        job_id = f"job_{timestamp}_{safe_slug}"

        # Initialize stage structures
        stages_init = []
        for s in STAGE_DEFINITIONS:
            stages_init.append({
                "num": s["num"],
                "name": s["name"],
                "model": s["model"],
                "desc": s["desc"],
                "status": "waiting",
                "percent": 0.0,
                "duration": 0.0,
                "message": "Waiting..."
            })

        job_data = {
            "id": job_id,
            "title": title or safe_slug,
            "source": source_path_or_url,
            "is_url": is_url,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "status": "queued",
            "progress": 0.0,
            "current_stage": "Queued",
            "stages": stages_init,
            "created_at": timestamp,
            "start_time": None,
            "elapsed_time": 0.0,
            "estimated_remaining": max(30.0, duration * 2.5) if duration > 0 else 60.0,
            "video_duration": duration,
            "logs": [f"[{time.strftime('%H:%M:%S')}] Job created and added to queue."],
            "error": None,
            "outputs": {},
            "segments": []
        }

        with self.lock:
            self.jobs[job_id] = job_data
            self.cancel_flags[job_id] = threading.Event()
            self.queue.append(job_id)
            self._save_job_history()

        logger.info(f"[JobManager] Created and queued job {job_id}")
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancels a queued or currently executing job."""
        with self.lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            if job["status"] == "queued":
                if job_id in self.queue:
                    self.queue.remove(job_id)
                job["status"] = "cancelled"
                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Job cancelled while in queue.")
                self._save_job_history()
                return True

            if job["status"] == "running":
                if job_id in self.cancel_flags:
                    self.cancel_flags[job_id].set()
                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Cancellation requested by user...")
                return True

        return False

    def resume_job(self, job_id: str) -> bool:
        """Re-enqueues an incomplete or cancelled job to resume from last checkpoint."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            job = self.jobs[job_id]
            if job["status"] in ("running", "queued"):
                return False

            job["status"] = "queued"
            job["error"] = None
            job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Job resumed from cached checkpoints.")
            self.cancel_flags[job_id] = threading.Event()
            self.queue.append(job_id)
            self._save_job_history()
            return True

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the current state of a job."""
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Returns all jobs ordered by creation time descending."""
        with self.lock:
            sorted_jobs = sorted(self.jobs.values(), key=lambda x: x.get("created_at", 0), reverse=True)
            return sorted_jobs

    def redub_line(self, job_id: str, segment_id: int, new_text: str, new_speaker: str) -> Dict[str, Any]:
        """Granularly re-dubs a single line in an already completed job."""
        job = self.get_job(job_id)
        if not job or job["status"] != "completed":
            raise ValueError("Job not found or not yet completed.")

        dubber = BhasharaDubber(job_id)
        proj = dubber.redub_single_line(segment_id, new_text=new_text, new_speaker=new_speaker, tts_engine="edge_tts")

        # Update exported files
        final_video_name = f"dubbed_{job_id}.mp4"
        final_video_path = OUTPUT_DIR / final_video_name
        if proj.final_dubbed_video_path and Path(proj.final_dubbed_video_path).exists():
            shutil.copy2(proj.final_dubbed_video_path, final_video_path)

        final_audio_path = OUTPUT_DIR / f"dubbed_{job_id}.wav"
        if proj.final_dubbed_audio_path and Path(proj.final_dubbed_audio_path).exists():
            shutil.copy2(proj.final_dubbed_audio_path, final_audio_path)

        out_stage9 = dubber.project_dir / "output"
        sub_map = {
            "original": out_stage9 / "dubbed.orig.srt",
            "translated": out_stage9 / "dubbed.translated.srt",
            "bilingual": out_stage9 / "dubbed.bilingual.srt"
        }
        for sub_key, src_file in sub_map.items():
            if src_file.exists():
                dest_sub = OUTPUT_DIR / f"{job_id}_{sub_key}.srt"
                shutil.copy2(src_file, dest_sub)

        # Update job segments in memory
        job["segments"] = [s.model_dump() for s in proj.segments]
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Re-dubbed Line #{segment_id} ('{new_text}'). Remux complete.")
        with self.lock:
            self._save_job_history()

        return {"success": True, "job": job}

    def get_disk_usage(self) -> Dict[str, Any]:
        """Calculates total disk space used by cache, uploads, and outputs."""
        def get_dir_size(p: Path) -> int:
            if not p.exists(): return 0
            total = 0
            for root, _, files in os.walk(p):
                for f in files:
                    try:
                        total += (Path(root) / f).stat().st_size
                    except Exception:
                        pass
            return total

        cache_bytes = get_dir_size(PROJECT_ROOT / "cache")
        upload_bytes = get_dir_size(UPLOADS_DIR)
        output_bytes = get_dir_size(OUTPUT_DIR)
        proof_bytes = get_dir_size(PROJECT_ROOT / "proof")

        total_bytes = cache_bytes + upload_bytes + output_bytes + proof_bytes
        return {
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "cache_mb": round(cache_bytes / (1024 * 1024), 2),
            "uploads_mb": round(upload_bytes / (1024 * 1024), 2),
            "output_mb": round(output_bytes / (1024 * 1024), 2),
            "proof_mb": round(proof_bytes / (1024 * 1024), 2),
        }

    def clean_temp_files(self) -> Dict[str, Any]:
        """Deletes intermediate cache files while preserving finished outputs."""
        cleaned_mb = 0.0
        cache_dir = PROJECT_ROOT / "cache"
        if cache_dir.exists():
            for item in cache_dir.iterdir():
                if item.is_dir() and item.name.startswith("job_"):
                    # Delete intermediate wav folders inside job cache
                    for sub in ["tts", "aligned", "stems"]:
                        sub_p = item / sub
                        if sub_p.exists():
                            sz = sum(f.stat().st_size for f in sub_p.glob("*") if f.is_file())
                            cleaned_mb += sz / (1024 * 1024)
                            shutil.rmtree(sub_p, ignore_errors=True)

        return {"cleaned_mb": round(cleaned_mb, 2)}

    def _worker_loop(self):
        """Worker loop processing jobs sequentially."""
        while True:
            job_id = None
            with self.lock:
                if self.queue:
                    job_id = self.queue.pop(0)
                    self.active_job_id = job_id
            
            if not job_id:
                time.sleep(0.5)
                continue

            try:
                self._execute_job(job_id)
            except Exception as e:
                logger.error(f"[JobManager] Error executing job {job_id}: {e}", exc_info=True)
            finally:
                with self.lock:
                    self.active_job_id = None
                time.sleep(0.5)

    def _execute_job(self, job_id: str):
        """Runs the 10-stage pipeline with live progress callbacks."""
        job = self.jobs[job_id]
        cancel_evt = self.cancel_flags.get(job_id)
        
        job["status"] = "running"
        job["start_time"] = time.time()
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Started execution of {job['title']}...")
        self._save_job_history()

        dubber = BhasharaDubber(job_id)
        project = dubber.get_or_create_project(
            source_video=job["source"],
            target_lang=job["target_lang"],
            source_lang=job["source_lang"],
            voice_mode="clone"
        )

        stage_start_times = {}

        def progress_cb(stage_num: int, stage_name: str, status: str, percent: float, msg: str):
            now = time.time()
            with self.lock:
                job["current_stage"] = f"Stage {stage_num}: {stage_name}"
                job["progress"] = min(99.0, max(job["progress"], percent))
                job["elapsed_time"] = round(now - job["start_time"], 1)

                # Estimate remaining time
                if job["progress"] > 5.0:
                    rate = job["elapsed_time"] / job["progress"]
                    job["estimated_remaining"] = max(0.0, round((100.0 - job["progress"]) * rate, 1))

                # Update stage record
                idx = stage_num - 1
                if 0 <= idx < len(job["stages"]):
                    st = job["stages"][idx]
                    st["status"] = status
                    st["message"] = msg
                    if status == "running" and stage_num not in stage_start_times:
                        stage_start_times[stage_num] = now
                    elif status == "done":
                        st0 = stage_start_times.get(stage_num, now)
                        st["duration"] = round(now - st0, 2)
                        st["percent"] = 100.0

                log_line = f"[{time.strftime('%H:%M:%S')}] [Stage {stage_num}] {msg}"
                job["logs"].append(log_line)

        try:
            # Run overlapped execution with cancellation check
            project = dubber.run_overlapped(
                project,
                tts_engine="edge_tts",
                enable_lipsync=False,
                progress_callback=progress_cb,
                cancel_check=lambda: cancel_evt.is_set() if cancel_evt else False
            )

            # Export final outputs to output/ folder
            stem_name = Path(project.source_video).stem
            final_video_name = f"dubbed_{job_id}.mp4"
            final_video_path = OUTPUT_DIR / final_video_name
            if project.final_dubbed_video_path and Path(project.final_dubbed_video_path).exists():
                shutil.copy2(project.final_dubbed_video_path, final_video_path)

            final_audio_path = OUTPUT_DIR / f"dubbed_{job_id}.wav"
            if project.final_dubbed_audio_path and Path(project.final_dubbed_audio_path).exists():
                shutil.copy2(project.final_dubbed_audio_path, final_audio_path)

            vocal_stem_path = OUTPUT_DIR / f"vocals_{job_id}.wav"
            if project.vocal_stem_path and Path(project.vocal_stem_path).exists():
                shutil.copy2(project.vocal_stem_path, vocal_stem_path)

            music_stem_path = OUTPUT_DIR / f"music_{job_id}.wav"
            if project.instrumental_stem_path and Path(project.instrumental_stem_path).exists():
                shutil.copy2(project.instrumental_stem_path, music_stem_path)

            # Subtitles
            sub_paths = {}
            out_stage9 = dubber.project_dir / "output"
            sub_map = {
                "original": out_stage9 / "dubbed.orig.srt",
                "translated": out_stage9 / "dubbed.translated.srt",
                "bilingual": out_stage9 / "dubbed.bilingual.srt"
            }
            for sub_key, src_file in sub_map.items():
                if src_file.exists():
                    dest_sub = OUTPUT_DIR / f"{job_id}_{sub_key}.srt"
                    shutil.copy2(src_file, dest_sub)
                    sub_paths[sub_key] = str(dest_sub)

            with self.lock:
                job["status"] = "completed"
                job["progress"] = 100.0
                job["estimated_remaining"] = 0.0
                job["elapsed_time"] = round(time.time() - job["start_time"], 1)
                job["outputs"] = {
                    "original_video": project.source_video,
                    "dubbed_video": str(final_video_path),
                    "dubbed_audio": str(final_audio_path),
                    "vocals_stem": str(vocal_stem_path),
                    "music_stem": str(music_stem_path),
                    "subtitles": sub_paths,
                    "segments_json": str(dubber.segments_file)
                }
                job["segments"] = [s.model_dump() for s in project.segments]
                for st in job["stages"]:
                    st["status"] = "done"
                    st["percent"] = 100.0

                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] 🎉 Dubbing complete! Output saved to {final_video_path}")
                self._save_job_history()

        except Exception as e:
            now = time.time()
            with self.lock:
                if cancel_evt and cancel_evt.is_set():
                    job["status"] = "cancelled"
                    job["logs"].append(f"[{time.strftime('%H:%M:%S')}] Job stopped cleanly by user.")
                else:
                    job["status"] = "failed"
                    job["error"] = str(e)
                    job["logs"].append(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")

                job["elapsed_time"] = round(now - job["start_time"], 1)
                self._save_job_history()
            VRAMJanitor.clean_vram("job_cleanup")
