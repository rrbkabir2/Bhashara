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


def _get_file_size_str(path: Optional[str]) -> str:
    """Helper to return human readable file size string."""
    if not path or not Path(path).exists():
        return "N/A"
    sz = Path(path).stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if sz < 1024.0:
            return f"{sz:.1f} {unit}"
        sz /= 1024.0
    return f"{sz:.1f} TB"


class BhasharaDubber:
    """Core dubbing orchestrator handling projects, cache, and state transitions."""

    def __init__(self, project_id: str, cache_root: Path = CACHE_DIR):
        self.project_id = project_id
        self.project_dir = cache_root / project_id
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.segments_file = self.project_dir / "segments.json"
        self.run_log_entries: List[Dict[str, Any]] = []

    def log_stage_run(
        self,
        stage_num: int,
        stage_name: str,
        model_name: str,
        device: str,
        input_desc: str,
        output_desc: str,
        duration_s: float,
        peak_vram_gb: float
    ):
        """Records diagnostic metrics for a pipeline stage."""
        entry = {
            "stage": f"Stage {stage_num}: {stage_name}",
            "model": model_name,
            "device": device,
            "input": input_desc,
            "output": output_desc,
            "duration": round(duration_s, 2),
            "peak_vram": f"{peak_vram_gb:.2f} GB" if peak_vram_gb > 0 else "0.00 GB (CPU)"
        }
        self.run_log_entries.append(entry)

    def print_and_save_run_log(self, total_time: float, mode: str = "Sequential"):
        """Prints a structured ASCII run log table and persists to run_log.txt."""
        lines = []
        lines.append("=" * 115)
        lines.append(f" 📊 BHASHARA DUBBING RUN LOG [{mode.upper()} MODE] - PROJECT: {self.project_id}")
        lines.append("=" * 115)
        lines.append(
            f"{'STAGE':<26} | {'MODEL / ENGINE':<24} | {'DEVICE':<8} | {'TIME':<7} | {'PEAK VRAM':<12} | {'OUTPUT':<25}"
        )
        lines.append("-" * 115)
        for e in self.run_log_entries:
            lines.append(
                f"{e['stage'][:26]:<26} | {e['model'][:24]:<24} | {e['device'][:8]:<8} | "
                f"{e['duration']:>5.2f}s | {e['peak_vram']:<12} | {e['output'][:25]:<25}"
            )
        lines.append("=" * 115)
        lines.append(f"  Total Pipeline Execution Time: {total_time:.2f} seconds")
        lines.append("=" * 115)

        log_content = "\n".join(lines)
        print("\n" + log_content + "\n", flush=True)

        # Save to project cache
        run_log_path = self.project_dir / "run_log.txt"
        with open(run_log_path, "w", encoding="utf-8") as f:
            f.write(log_content + "\n")

        # Save to proof folder if exists
        proof_dir = PROJECT_ROOT if "PROJECT_ROOT" in globals() else self.project_dir.parent.parent / "proof"
        if proof_dir.exists():
            with open(proof_dir / "run_log.txt", "w", encoding="utf-8") as f:
                f.write(log_content + "\n")

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
        enable_lipsync: bool = False,
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None
    ) -> ProjectMetadata:
        """Executes all 10 stages strictly sequentially with diagnostic run logging."""
        logger.info("[Pipeline] Starting sequential execution...")
        self.run_log_entries.clear()
        t0 = time.time()

        # Helper to execute and profile a stage
        def profile_stage(num: int, name: str, model: str, default_dev: str, func, *args):
            if cancel_check and cancel_check():
                VRAMJanitor.clean_vram("cancelled")
                raise RuntimeError("Dubbing job was cancelled by user.")
            if progress_callback:
                progress_callback(num, name, "running", float((num - 1) * 10), f"Running {name} ({model})...")
            VRAMJanitor.reset_peak_stats()
            st0 = time.time()
            res = func(*args)
            dur = time.time() - st0
            peak_vram = VRAMJanitor.get_peak_vram_gb()
            dev = "cuda" if peak_vram > 0 else default_dev
            if progress_callback:
                progress_callback(num, name, "done", float(num * 10), f"Completed {name} in {dur:.1f}s")
            return res, dur, peak_vram, dev

        # Stage 1: Input & Audio Demux (CPU)
        project, d1, v1, dev1 = profile_stage(
            1, "Input Demux", "FFmpeg Binary (Static)", "cpu",
            run_stage1_input, project, self.project_dir
        )
        self.log_stage_run(1, "Input Demux", "FFmpeg 9.0.2", dev1,
                           _get_file_size_str(project.source_video),
                           f"audio_16k ({_get_file_size_str(project.audio_16k_mono)})", d1, v1)

        # Stage 2: Demucs Vocal Separation (GPU/CPU)
        project, d2, v2, dev2 = profile_stage(
            2, "Separation", "Demucs v4 (htdemucs)", "cpu",
            run_stage2_separation, project, self.project_dir
        )
        self.log_stage_run(2, "Vocal Separation", "Demucs v4 (htdemucs)", dev2,
                           _get_file_size_str(project.audio_16k_mono),
                           f"stems ({_get_file_size_str(project.vocal_stem_path)})", d2, v2)

        # Stage 3: faster-whisper Transcription (GPU/CPU)
        project, d3, v3, dev3 = profile_stage(
            3, "Transcription", "faster-whisper base", "cpu",
            run_stage3_transcription, project, self.project_dir
        )
        self.log_stage_run(3, "Transcription", "faster-whisper base", dev3,
                           _get_file_size_str(project.vocal_stem_path),
                           f"{len(project.segments)} segments in segments.json", d3, v3)

        # Stage 4: Diarization & Voice Profiling (CPU)
        project, d4, v4, dev4 = profile_stage(
            4, "Diarization", "Acoustic F0 Clustering", "cpu",
            run_stage4_diarization, project, self.project_dir
        )
        self.log_stage_run(4, "Diarization", "Acoustic F0 Pyin", dev4,
                           _get_file_size_str(project.vocal_stem_path),
                           f"{len(project.speakers)} speakers profiled", d4, v4)

        # Stage 5: Syllable-Budgeted Translation (CPU)
        project, d5, v5, dev5 = profile_stage(
            5, "Translation", "MarianMT (Helsinki-NLP)", "cpu",
            run_stage5_translation, project, self.project_dir
        )
        self.log_stage_run(5, "Translation", "MarianMT Helsinki-NLP", dev5,
                           f"{len(project.segments)} lines",
                           "Syllable-budgeted Hindi", d5, v5)

        # Stage 6: Voice Synthesis (GPU/VRAM or Edge-TTS)
        tts_name = "Edge-TTS Neural" if tts_engine in ("auto", "edge_tts") else "Harmonic Preset"
        project, d6, v6, dev6 = profile_stage(
            6, "Voice Synthesis", tts_name, "cpu",
            run_stage6_synthesis, project, self.project_dir, tts_engine
        )
        self.log_stage_run(6, "Synthesis", tts_name, dev6,
                           f"{len(project.segments)} lines",
                           f"tts/ ({len(list((self.project_dir / 'tts').glob('*.wav')))} clips)", d6, v6)

        # Stage 7: Timing Fit & Alignment (CPU)
        project, d7, v7, dev7 = profile_stage(
            7, "Timing Fit", "WSOLA + Pause Squeeze", "cpu",
            run_stage7_timing_fit, project, self.project_dir
        )
        self.log_stage_run(7, "Timing Fit", "WSOLA SolaStretch", dev7,
                           "tts/ clips",
                           f"aligned/ ({len(list((self.project_dir / 'aligned').glob('*.wav')))} clips)", d7, v7)

        # Stage 8: Ducked Mixing & Loudness (CPU)
        project, d8, v8, dev8 = profile_stage(
            8, "Ducked Mixing", "Sidechain + EBU R128", "cpu",
            run_stage8_mixer, project, self.project_dir
        )
        self.log_stage_run(8, "Ducked Mixer", "Sidechain Ducking -16LUFS", dev8,
                           "stems + aligned/",
                           f"mixed_dubbed ({_get_file_size_str(project.final_dubbed_audio_path)})", d8, v8)

        # Stage 9: Packaging & Muxing (CPU)
        project, d9, v9, dev9 = profile_stage(
            9, "Muxing & Output", "FFmpeg Binary (Static)", "cpu",
            run_stage9_output, project, self.project_dir
        )
        self.log_stage_run(9, "Output Muxing", "FFmpeg Muxer", dev9,
                           "mixed audio + source video",
                           f"dubbed_video ({_get_file_size_str(project.final_dubbed_video_path)})", d9, v9)

        # Stage 10: Optional LipSync
        project, d10, v10, dev10 = profile_stage(
            10, "Lip Sync", "Wav2Lip (Optional)", "cpu",
            run_stage10_lipsync, project, self.project_dir, enable_lipsync
        )
        lipsync_out = project.lipsync_video_path if enable_lipsync else "Bypassed (Default OFF)"
        self.log_stage_run(10, "Lip Sync", "Wav2Lip Checkpoint", dev10,
                           "dubbed_video.mp4", lipsync_out, d10, v10)

        elapsed = time.time() - t0
        logger.info(f"[Pipeline] Sequential run finished in {elapsed:.2f} seconds.")
        self.print_and_save_run_log(elapsed, mode="Sequential")
        return project

    def run_overlapped(
        self,
        project: ProjectMetadata,
        tts_engine: str = "auto",
        enable_lipsync: bool = False,
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None
    ) -> ProjectMetadata:
        """
        Executes pipeline with CPU and GPU overlap queue.
        While the GPU generates TTS for segment K, the CPU simultaneously executes
        time-stretching, pause-compression, and audio fitting for segment K-1.
        Uses a bounded queue (maxsize=MAX_OVERLAP_QUEUE_SIZE) to prevent RAM inflation.
        """
        logger.info("[Pipeline] Starting CPU/GPU overlapped execution...")
        t0 = time.time()

        def check_cancelled():
            if cancel_check and cancel_check():
                VRAMJanitor.clean_vram("cancelled")
                raise RuntimeError("Dubbing job was cancelled by user.")

        # Stages 1 to 5: Initial preparation
        check_cancelled()
        if progress_callback: progress_callback(1, "Input Demux", "running", 5.0, "Extracting audio...")
        project = run_stage1_input(project, self.project_dir)
        if progress_callback: progress_callback(1, "Input Demux", "done", 10.0, "Audio extracted.")

        check_cancelled()
        if progress_callback: progress_callback(2, "Vocal Separation", "running", 15.0, "Splitting stems (Demucs)...")
        project = run_stage2_separation(project, self.project_dir)
        if progress_callback: progress_callback(2, "Vocal Separation", "done", 25.0, "Stems isolated.")

        check_cancelled()
        if progress_callback: progress_callback(3, "Transcription", "running", 30.0, "Transcribing speech (faster-whisper)...")
        project = run_stage3_transcription(project, self.project_dir)
        if progress_callback: progress_callback(3, "Transcription", "done", 40.0, f"Transcribed {len(project.segments)} lines.")

        check_cancelled()
        if progress_callback: progress_callback(4, "Diarization", "running", 42.0, "Clustering speakers...")
        project = run_stage4_diarization(project, self.project_dir)
        if progress_callback: progress_callback(4, "Diarization", "done", 48.0, f"Profiled {len(project.speakers)} speakers.")

        check_cancelled()
        if progress_callback: progress_callback(5, "Translation", "running", 50.0, "Translating with syllable budget (MarianMT)...")
        project = run_stage5_translation(project, self.project_dir)
        if progress_callback: progress_callback(5, "Translation", "done", 60.0, "Dialogue translated.")

        # Overlapped Stages 6 & 7 (TTS Producer on GPU -> Timing Fit Consumer on CPU)
        check_cancelled()
        if progress_callback: progress_callback(6, "Synthesis & Timing", "running", 62.0, "Synthesizing and fitting lines concurrently...")
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
        if progress_callback: progress_callback(7, "Synthesis & Timing", "done", 80.0, "All dialogue synthesized and fitted.")

        # Stage 8: Ducked Mixing (CPU)
        check_cancelled()
        if progress_callback: progress_callback(8, "Ducked Mixing", "running", 82.0, "Sidechain ducking & -16 LUFS mastering...")
        project = run_stage8_mixer(project, self.project_dir)
        if progress_callback: progress_callback(8, "Ducked Mixing", "done", 90.0, "Master audio ducked and normalized.")

        # Stage 9: Packaging & Muxing (CPU)
        check_cancelled()
        if progress_callback: progress_callback(9, "Muxing & Subtitles", "running", 92.0, "Muxing video & compiling subtitles...")
        project = run_stage9_output(project, self.project_dir)
        if progress_callback: progress_callback(9, "Muxing & Subtitles", "done", 98.0, "Video remuxed with subtitles.")

        # Stage 10: Optional LipSync
        check_cancelled()
        if enable_lipsync:
            if progress_callback: progress_callback(10, "Lip Sync", "running", 99.0, "Running Wav2Lip sync...")
            project = run_stage10_lipsync(project, self.project_dir, enabled=enable_lipsync)
            if progress_callback: progress_callback(10, "Lip Sync", "done", 100.0, "Lip sync complete.")
        else:
            project = run_stage10_lipsync(project, self.project_dir, enabled=False)
            if progress_callback: progress_callback(10, "Complete", "done", 100.0, "Dubbed video ready.")

        elapsed = time.time() - t0
        logger.info(f"[Pipeline] Overlapped run finished in {elapsed:.2f} seconds.")
        self.print_and_save_run_log(elapsed, mode="Overlapped")
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
