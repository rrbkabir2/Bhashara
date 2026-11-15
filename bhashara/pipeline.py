"""
Bhashara Pipeline Orchestrator & Concurrency Engine
Coordinates the 10-stage video dubbing pipeline.
Features:
- Resumable execution based on segments.json single source of truth
- Granular single-line re-dubbing
- CPU/GPU Overlap Queue: Asynchronous producer-consumer overlap between GPU model
  inference and CPU DSP/formatting tasks with small bounded queue.
- Honest speedup benchmarking (sequential vs overlapped).
"""

import time
import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from bhashara.models.schema import ProjectMetadata, Segment
from bhashara.config import CACHE_DIR, MAX_OVERLAP_QUEUE_SIZE
from bhashara.vram_manager import VRAMJanitor

from bhashara.stages.stage1_input import run_stage1_input
from bhashara.stages.stage2_separation import run_stage2_separation
from bhashara.stages.stage3_transcribe import run_stage3_transcription
from bhashara.stages.stage4_diarize import run_stage4_diarization
from bhashara.stages.stage5_translate import run_stage5_translation
from bhashara.stages.stage6_synthesis import run_stage6_synthesis, HarmonicPresetAdapter, EdgeTTSAdapter
from bhashara.stages.stage7_timing import run_stage7_timing_fit
from bhashara.stages.stage8_mixer import run_stage8_mixer
from bhashara.stages.stage9_output import run_stage9_output
from bhashara.stages.stage10_lipsync import run_stage10_lipsync

logger = logging.getLogger("bhashara.pipeline")


class BhasharaDubber:
    """Core dubbing orchestrator handling projects, cache, and state transitions."""

    def __init__(self, project_id: str, cache_root: Path = CACHE_DIR):
        self.project_id = project_id
        self.project_dir = cache_root / project_id
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.segments_file = self.project_dir / "segments.json"

    def get_or_create_project(
        self,
        source_video: str,
        target_lang: str = "hi",
        source_lang: str = "en",
        voice_mode: str = "clone"
    ) -> ProjectMetadata:
        """Loads cached segments.json if it exists, otherwise initializes a new project."""
        if self.segments_file.exists():
            try:
                project = ProjectMetadata.load(self.segments_file)
                logger.info(f"Loaded existing project '{self.project_id}' from {self.segments_file}")
                return project
            except Exception as e:
                logger.warning(f"Could not load corrupted segments.json ({e}). Creating fresh.")

        project = ProjectMetadata(
            project_id=self.project_id,
            source_video=source_video,
            source_language=source_lang,
            target_language=target_lang,
            voice_mode=voice_mode
        )
        project.save(self.segments_file)
        return project

    def run_sequential(
        self,
        project: ProjectMetadata,
        tts_engine: str = "auto",
        enable_lipsync: bool = False
    ) -> ProjectMetadata:
        """Executes all 10 stages strictly sequentially."""
        logger.info("[Pipeline] Starting sequential execution...")
        t0 = time.time()

        # Stage 1: Input & Audio Demux (CPU)
        project = run_stage1_input(project, self.project_dir)

        # Stage 2: Demucs Vocal Separation (GPU)
        project = run_stage2_separation(project, self.project_dir)

        # Stage 3: faster-whisper Transcription (GPU)
        project = run_stage3_transcription(project, self.project_dir)

        # Stage 4: Diarization & Voice Profiling (GPU/CPU)
        project = run_stage4_diarization(project, self.project_dir)

        # Stage 5: Syllable-Budgeted Translation (CPU)
        project = run_stage5_translation(project, self.project_dir)

        # Stage 6: Voice Synthesis (GPU/VRAM)
        project = run_stage6_synthesis(project, self.project_dir, preferred_engine=tts_engine)

        # Stage 7: Timing Fit & Alignment (CPU)
        project = run_stage7_timing_fit(project, self.project_dir)

        # Stage 8: Ducked Mixing & Loudness (CPU)
        project = run_stage8_mixer(project, self.project_dir)

        # Stage 9: Packaging & Muxing (CPU)
        project = run_stage9_output(project, self.project_dir)

        # Stage 10: Optional LipSync
        project = run_stage10_lipsync(project, self.project_dir, enabled=enable_lipsync)

        elapsed = time.time() - t0
        logger.info(f"[Pipeline] Sequential run finished in {elapsed:.2f} seconds.")
        return project

    def run_overlapped(
        self,
        project: ProjectMetadata,
        tts_engine: str = "auto",
        enable_lipsync: bool = False
    ) -> ProjectMetadata:
        """
        Executes pipeline with CPU and GPU overlap queue.
        While the GPU generates TTS for segment K, the CPU simultaneously executes
        time-stretching, pause-compression, and audio fitting for segment K-1.
        Uses a bounded queue (maxsize=MAX_OVERLAP_QUEUE_SIZE) to prevent RAM inflation.
        """
        logger.info("[Pipeline] Starting CPU/GPU overlapped execution...")
        t0 = time.time()

        # Stages 1 to 5: Initial preparation
        project = run_stage1_input(project, self.project_dir)
        project = run_stage2_separation(project, self.project_dir)
        project = run_stage3_transcription(project, self.project_dir)
        project = run_stage4_diarization(project, self.project_dir)
        project = run_stage5_translation(project, self.project_dir)

        # Overlapped Stages 6 & 7 (TTS Producer on GPU -> Timing Fit Consumer on CPU)
        tts_dir = self.project_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir = self.project_dir / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        transfer_queue: queue.Queue = queue.Queue(maxsize=MAX_OVERLAP_QUEUE_SIZE)
        stop_token = object()
        consumer_errors = []

        # Select TTS adapter
        if tts_engine == "edge_tts":
            adapter = EdgeTTSAdapter()
        else:
            adapter = HarmonicPresetAdapter()

        def cpu_consumer_worker():
            """CPU worker concurrently fitting audio as soon as TTS completes a segment."""
            from bhashara.utils.audio_utils import load_audio, save_audio, compress_pauses, time_stretch_sola
            from bhashara.config import MIN_SPEEDDOWN_FACTOR, MAX_SPEEDUP_FACTOR
            import numpy as np

            while True:
                item = transfer_queue.get()
                if item is stop_token:
                    transfer_queue.task_done()
                    break
                try:
                    seg, raw_wav = item
                    audio, sr = load_audio(Path(raw_wav))
                    curr_dur = len(audio) / sr
                    target_dur = seg.target_duration
                    
                    # 1. Micro-pause compression
                    if curr_dur > target_dur:
                        audio = compress_pauses(audio, sr, max_silence_sec=0.08)
                        curr_dur = len(audio) / sr

                    # 2. Time-stretch
                    speed_ratio = curr_dur / target_dur
                    clamped_speed = np.clip(speed_ratio, MIN_SPEEDDOWN_FACTOR, MAX_SPEEDUP_FACTOR)
                    if abs(clamped_speed - 1.0) > 0.02:
                        audio = time_stretch_sola(audio, sr, float(clamped_speed))
                        seg.speed_factor = round(float(clamped_speed), 2)

                    # 3. Save aligned clip
                    out_wav = aligned_dir / f"seg_{seg.id:04d}.wav"
                    save_audio(out_wav, audio, sr)
                    seg.fitted_audio_path = str(out_wav)
                    seg.status = "completed"
                    logger.debug(f"[Overlap Consumer] Processed Seg #{seg.id} on CPU in parallel.")
                except Exception as ex:
                    consumer_errors.append(ex)
                finally:
                    transfer_queue.task_done()

        # Start CPU consumer thread
        consumer_thread = threading.Thread(target=cpu_consumer_worker, daemon=True)
        consumer_thread.start()

        # GPU Producer: Synthesizes lines and immediately enqueues for CPU worker
        for seg in project.segments:
            seg_wav = tts_dir / f"seg_{seg.id:04d}.wav"
            spk = project.speakers.get(seg.speaker)
            ref_wav = spk.ref_audio_path if spk else None
            gender = spk.gender if spk else "neutral"

            # GPU Model synthesis
            adapter.synthesize(
                text=seg.translated_text,
                lang=project.target_language,
                speaker_gender=gender,
                ref_audio_path=ref_wav,
                output_path=seg_wav
            )
            seg.raw_tts_path = str(seg_wav)
            seg.dirty = False
            
            # Feed into bounded queue for concurrent CPU processing
            transfer_queue.put((seg, str(seg_wav)))

        # Signal completion and wait for queue to drain
        transfer_queue.put(stop_token)
        transfer_queue.join()
        consumer_thread.join()

        if consumer_errors:
            raise RuntimeError(f"Errors occurred in overlapped CPU consumer: {consumer_errors}")

        project.stage_status["synthesis"] = "completed"
        project.stage_status["timing"] = "completed"
        project.save(self.segments_file)

        # Stage 8: Ducked Mixing (CPU)
        project = run_stage8_mixer(project, self.project_dir)

        # Stage 9: Packaging & Muxing (CPU)
        project = run_stage9_output(project, self.project_dir)

        # Stage 10: Optional LipSync
        project = run_stage10_lipsync(project, self.project_dir, enabled=enable_lipsync)

        elapsed = time.time() - t0
        logger.info(f"[Pipeline] Overlapped run finished in {elapsed:.2f} seconds.")
        return project

    def redub_single_line(
        self,
        segment_id: int,
        new_text: Optional[str] = None,
        new_speaker: Optional[str] = None,
        tts_engine: str = "auto"
    ) -> ProjectMetadata:
        """
        Granular Single-Line Regeneration.
        Only re-synthesizes and re-fits the modified segment, then remuxes the master audio
        without touching any other lines.
        """
        project = ProjectMetadata.load(self.segments_file)
        
        target_seg = None
        for seg in project.segments:
            if seg.id == segment_id:
                target_seg = seg
                break
                
        if not target_seg:
            raise ValueError(f"Segment ID {segment_id} not found in project.")

        logger.info(f"[Re-dub] Regenerating single line #{segment_id}...")
        
        if new_text is not None:
            target_seg.translated_text = new_text
        if new_speaker is not None:
            target_seg.speaker = new_speaker
            
        target_seg.dirty = True
        target_seg.status = "pending"
        project.save(self.segments_file)

        # Re-run only Synthesis, Timing, Mixer, and Output
        project = run_stage6_synthesis(project, self.project_dir, preferred_engine=tts_engine)
        project = run_stage7_timing_fit(project, self.project_dir)
        project = run_stage8_mixer(project, self.project_dir)
        project = run_stage9_output(project, self.project_dir)

        logger.info(f"[Re-dub] Line #{segment_id} regenerated and remuxed successfully!")
        return project
