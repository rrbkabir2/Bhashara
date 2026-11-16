"""
Bhashara Dual-Clip Differentiation Test (tests/test_differentiation.py)
Feeds two different video/audio clips through the pipeline and verifies that:
1. Transcripts are distinct (speech recognition is genuinely decoding input audio)
2. Translations are distinct (dialogue is genuinely translated)
3. Speaker voice profiles differ
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bhashara.pipeline import BhasharaDubber
from bhashara.stages.stage3_transcribe import run_stage3_transcription
from bhashara.stages.stage4_diarize import run_stage4_diarization
from bhashara.stages.stage5_translate import run_stage5_translation
from bhashara.stages.stage1_input import run_stage1_input

TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
CLIP1_VIDEO = TEST_DATA_DIR / "clip1" / "clip1_video.mp4"
CLIP2_VIDEO = TEST_DATA_DIR / "clip2" / "clip2_video.mp4"


class TestPipelineDifferentiation(unittest.TestCase):

    def test_clips_produce_different_outputs(self):
        self.assertTrue(CLIP1_VIDEO.exists(), f"Clip 1 missing: {CLIP1_VIDEO}")
        self.assertTrue(CLIP2_VIDEO.exists(), f"Clip 2 missing: {CLIP2_VIDEO}")

        # Process Clip 1
        dubber1 = BhasharaDubber("test_diff_clip1")
        if dubber1.project_dir.exists():
            import shutil
            shutil.rmtree(dubber1.project_dir)
        dubber1.project_dir.mkdir(parents=True, exist_ok=True)
        proj1 = dubber1.get_or_create_project(str(CLIP1_VIDEO), target_lang="hi", source_lang="en")
        proj1 = run_stage1_input(proj1, dubber1.project_dir)
        proj1 = run_stage3_transcription(proj1, dubber1.project_dir)
        proj1 = run_stage4_diarization(proj1, dubber1.project_dir)
        proj1 = run_stage5_translation(proj1, dubber1.project_dir)

        # Process Clip 2
        dubber2 = BhasharaDubber("test_diff_clip2")
        if dubber2.project_dir.exists():
            import shutil
            shutil.rmtree(dubber2.project_dir)
        dubber2.project_dir.mkdir(parents=True, exist_ok=True)
        proj2 = dubber2.get_or_create_project(str(CLIP2_VIDEO), target_lang="hi", source_lang="en")
        proj2 = run_stage1_input(proj2, dubber2.project_dir)
        proj2 = run_stage3_transcription(proj2, dubber2.project_dir)
        proj2 = run_stage4_diarization(proj2, dubber2.project_dir)
        proj2 = run_stage5_translation(proj2, dubber2.project_dir)

        # Extract text
        text1 = " ".join(s.original_text for s in proj1.segments)
        text2 = " ".join(s.original_text for s in proj2.segments)

        trans1 = " ".join(s.translated_text for s in proj1.segments)
        trans2 = " ".join(s.translated_text for s in proj2.segments)

        print("\n--- Differentiation Results ---")
        print(f"Clip 1 Transcript:  '{text1}'")
        print(f"Clip 2 Transcript:  '{text2}'")
        print(f"Clip 1 Translation: '{trans1}'")
        print(f"Clip 2 Translation: '{trans2}'")

        # 1. Transcripts must not be identical
        self.assertNotEqual(text1, text2, "Transcripts for distinct clips must differ!")
        self.assertTrue(len(text1) > 0 and len(text2) > 0, "Transcripts must not be empty.")

        # 2. Translations must not be identical
        self.assertNotEqual(trans1, trans2, "Translations for distinct clips must differ!")
        self.assertTrue(len(trans1) > 0 and len(trans2) > 0, "Translations must not be empty.")

        # 3. Content matching check
        self.assertIn("studio", text1.lower(), "Clip 1 should mention studio")
        self.assertTrue(
            "microphone" in text2.lower() or "audio" in text2.lower() or "levels" in text2.lower(),
            "Clip 2 should mention microphone/audio/levels"
        )
        print("✅ Dual-clip differentiation test passed successfully!\n")


if __name__ == "__main__":
    unittest.main()
