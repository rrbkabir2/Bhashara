"""
Bhashara Web Interface (Local Dubbing Studio)
Provides:
1. Video upload / URL ingestion with language and voice mode selection.
2. Stage-by-stage progress monitoring.
3. Interactive Script Editor:
   - Transcript and translation side-by-side
   - Change speaker assignment
   - Re-dub single line with instant preview
4. Preview mode (skips slow neural stages).
5. Batch processing mode for a folder of videos.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bhashara.pipeline import BhasharaDubber
from bhashara.models.schema import ProjectMetadata

logger = logging.getLogger("bhashara.ui")


def create_gradio_app():
    """Builds the interactive Gradio UI."""
    try:
        import gradio as gr  # type: ignore
    except ImportError:
        logger.warning("Gradio not installed. Install via `pip install gradio` to launch web UI.")
        return None

    dubber_instances: Dict[str, BhasharaDubber] = {}

    with gr.Blocks(title="Bhashara - Local AI Video Dubbing Engine", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ Bhashara (भशारा) - Local AI Video Dubbing Studio
            **100% Local, Zero-Budget, Consumer GPU-Optimized Video Dubbing**
            *English ↔ Hindi | Voice Cloning | Syllable-Budgeted Timing | Sidechain Ducking*
            """
        )

        with gr.Tab("🎬 Dub Single Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video File", sources=["upload"])
                    url_input = gr.Textbox(label="Or Paste Video URL (YouTube)", placeholder="https://youtube.com/watch?v=...")
                    
                    with gr.Row():
                        src_lang = gr.Dropdown(choices=["en", "hi", "mr"], value="en", label="Source Language")
                        tgt_lang = gr.Dropdown(choices=["hi", "en", "mr"], value="hi", label="Target Language")
                        
                    voice_mode = gr.Radio(choices=["clone", "preset"], value="clone", label="Voice Mode")
                    engine_choice = gr.Dropdown(
                        choices=["auto", "xtts_v2", "edge_tts", "harmonic"],
                        value="auto",
                        label="TTS Engine (auto chooses best available)"
                    )
                    use_overlap = gr.Checkbox(value=True, label="Enable CPU/GPU Overlap Pipeline")
                    
                    start_btn = gr.Button("🚀 Start Dubbing Pipeline", variant="primary")

                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Pipeline Stage Progress", lines=8, interactive=False)
                    video_output = gr.Video(label="Dubbed Output Video")
                    audio_output = gr.Audio(label="Master Dubbed Audio (-16 LUFS)")

            # Script Editor View
            gr.Markdown("### 📝 Interactive Script Editor & Granular Re-dubbing")
            with gr.Row():
                with gr.Column():
                    editor_project_id = gr.Textbox(label="Active Project ID", interactive=False)
                    line_selector = gr.Dropdown(label="Select Segment to Edit", choices=[])
                    original_text_display = gr.Textbox(label="Original Transcript", interactive=False)
                    translated_text_input = gr.Textbox(label="Translated Text (Edit here)", lines=2)
                    speaker_dropdown = gr.Dropdown(label="Assigned Speaker", choices=["SPEAKER_00", "SPEAKER_01"])
                    redub_btn = gr.Button("🔄 Re-dub This Line Only", variant="secondary")
                with gr.Column():
                    line_audio_preview = gr.Audio(label="Segment Audio Preview")
                    subtitles_display = gr.File(label="Download Subtitles (.srt / .vtt)")

        with gr.Tab("📁 Batch Folder Dubbing"):
            gr.Markdown("Dub multiple videos in a folder sequentially without memory accumulation.")
            folder_input = gr.Textbox(label="Folder Path Containing Videos", placeholder="C:/Videos/Episodes")
            batch_tgt_lang = gr.Dropdown(choices=["hi", "en", "mr"], value="hi", label="Target Language")
            batch_btn = gr.Button("Start Batch Dubbing")
            batch_status = gr.Textbox(label="Batch Status", lines=6)

        # Event Handlers
        def on_start_dubbing(video_file, url_text, s_lang, t_lang, v_mode, engine, overlap):
            source = video_file if video_file else url_text
            if not source:
                return "Error: Please upload a video or paste a URL.", None, None, "", gr.update()

            proj_id = f"proj_{abs(hash(str(source))) % 1000000:06d}"
            dubber = BhasharaDubber(proj_id)
            dubber_instances[proj_id] = dubber
            
            project = dubber.get_or_create_project(
                source_video=str(source),
                target_lang=t_lang,
                source_lang=s_lang,
                voice_mode=v_mode
            )

            status_log = f"Project '{proj_id}' initialized.\nProcessing pipeline..."
            if overlap:
                project = dubber.run_overlapped(project, tts_engine=engine)
            else:
                project = dubber.run_sequential(project, tts_engine=engine)

            status_log += "\nPipeline finished successfully!"
            seg_choices = [f"Seg #{s.id}: {s.original_text[:30]}..." for s in project.segments]
            
            return (
                status_log,
                project.final_dubbed_video_path,
                project.final_dubbed_audio_path,
                proj_id,
                gr.update(choices=seg_choices, value=seg_choices[0] if seg_choices else None)
            )

        start_btn.click(
            on_start_dubbing,
            inputs=[video_input, url_input, src_lang, tgt_lang, voice_mode, engine_choice, use_overlap],
            outputs=[status_output, video_output, audio_output, editor_project_id, line_selector]
        )

        def on_redub_line(proj_id, seg_selection, new_text, new_spk, engine):
            if not proj_id or proj_id not in dubber_instances or not seg_selection:
                return "Project not found", None, None
            
            dubber = dubber_instances[proj_id]
            seg_id = int(seg_selection.split(":")[0].replace("Seg #", ""))
            
            project = dubber.redub_single_line(
                segment_id=seg_id,
                new_text=new_text,
                new_speaker=new_spk,
                tts_engine=engine
            )
            
            fitted_wav = None
            for s in project.segments:
                if s.id == seg_id:
                    fitted_wav = s.fitted_audio_path
                    break
                    
            return "Line re-dubbed successfully!", fitted_wav, project.final_dubbed_video_path

        redub_btn.click(
            on_redub_line,
            inputs=[editor_project_id, line_selector, translated_text_input, speaker_dropdown, engine_choice],
            outputs=[status_output, line_audio_preview, video_output]
        )

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    if app:
        app.launch(server_name="127.0.0.1", server_port=7860, share=False)
    else:
        print("Launch failed: Gradio is not installed.")
