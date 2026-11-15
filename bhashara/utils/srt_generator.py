"""
Bhashara Subtitle Generator
Generates standard .srt and .vtt subtitle files:
- Original Language
- Translated Target Language
- Bilingual Subtitles (Translated on top, Original below)
"""

from pathlib import Path
from typing import List
from bhashara.models.schema import Segment


def format_srt_time(seconds: float) -> str:
    """Converts seconds float to SRT timestamp HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    """Converts seconds float to WebVTT timestamp HH:MM:SS.mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def generate_subtitles(
    segments: List[Segment],
    output_dir: Path,
    base_name: str = "dubbed"
) -> dict:
    """
    Writes original, translated, and bilingual .srt and .vtt files.
    Returns a dict with paths to all generated subtitle files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orig_srt = output_dir / f"{base_name}.orig.srt"
    trans_srt = output_dir / f"{base_name}.translated.srt"
    bilingual_srt = output_dir / f"{base_name}.bilingual.srt"
    
    with open(orig_srt, "w", encoding="utf-8") as f_orig, \
         open(trans_srt, "w", encoding="utf-8") as f_trans, \
         open(bilingual_srt, "w", encoding="utf-8") as f_bi:
         
        for i, seg in enumerate(segments, 1):
            t_start = format_srt_time(seg.start)
            t_end = format_srt_time(seg.end)
            
            # Original
            f_orig.write(f"{i}\n{t_start} --> {t_end}\n{seg.original_text.strip()}\n\n")
            
            # Translated
            f_trans.write(f"{i}\n{t_start} --> {t_end}\n{seg.translated_text.strip()}\n\n")
            
            # Bilingual (Translated top, Original bottom)
            f_bi.write(
                f"{i}\n{t_start} --> {t_end}\n"
                f"{seg.translated_text.strip()}\n"
                f"({seg.original_text.strip()})\n\n"
            )
            
    return {
        "original_srt": str(orig_srt),
        "translated_srt": str(trans_srt),
        "bilingual_srt": str(bilingual_srt)
    }
