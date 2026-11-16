"""
Test end-to-end execution of a job via JobManager and test single-line redubbing.
"""

import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.job_manager import JobManager

def test_full_pipeline_via_job_manager():
    jm = JobManager()
    clip_path = Path("tests/test_data/clip1/clip1_video.mp4").resolve()
    assert clip_path.exists(), "Test clip not found"
    
    print(f"[Test] Starting test job on {clip_path}...")
    job_id = jm.create_job(
        source_path_or_url=str(clip_path),
        is_url=False,
        target_lang="hi",
        source_lang="en",
        title="clip1_integration_test",
        duration=10.0
    )
    
    print(f"[Test] Job created with ID: {job_id}. Waiting for completion...")
    start_time = time.time()
    while True:
        job = jm.get_job(job_id)
        status = job.get("status")
        progress = job.get("progress", 0.0)
        curr_stage = job.get("current_stage")
        print(f"Status: {status} | Progress: {progress:.1f}% | Stage: {curr_stage}")
        
        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(2.0)
        
    print(f"[Test] Job finished with status '{status}' in {time.time() - start_time:.1f}s.")
    assert status == "completed", f"Job failed: {job.get('error')}"
    
    # Verify outputs
    outputs = job.get("outputs", {})
    print(f"[Test] Outputs: {outputs}")
    assert Path(outputs["dubbed_video"]).exists(), "Dubbed video missing"
    assert Path(outputs["dubbed_audio"]).exists(), "Dubbed audio missing"
    assert Path(outputs["vocals_stem"]).exists(), "Vocals stem missing"
    assert Path(outputs["music_stem"]).exists(), "Music stem missing"
    assert "original" in outputs["subtitles"]
    assert "translated" in outputs["subtitles"]
    assert "bilingual" in outputs["subtitles"]
    
    # Test single line redub
    print("\n[Test] Testing single-line re-dubbing...")
    segments = job.get("segments", [])
    if segments:
        first_seg_id = segments[0]["id"]
        res = jm.redub_line(
            job_id=job_id,
            segment_id=first_seg_id,
            new_text="नमस्ते दुनिया, यह एक नया अनुवाद है।",
            new_speaker="SPEAKER_00"
        )
        assert res["success"] is True
        print(f"[Test] Successfully re-dubbed line #{first_seg_id}!")
    
    print("\n[Test] End-to-end Web Studio pipeline test passed perfectly!\n")

if __name__ == "__main__":
    test_full_pipeline_via_job_manager()
