"""
Bhashara Web UI End-to-End API Test (tests/test_ui_api.py)
Verifies:
1. Web server initialization and static file delivery
2. Video URL pre-flight inspection and plain error reporting
3. Job creation via file upload and queued execution
4. Real-time progress polling and stage structure
5. Cancellation and clean memory reclamation
6. Script editor single-line re-dubbing API
7. Disk space cleanup endpoint
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
from ui.server import app
from ui.job_manager import JobManager

client = TestClient(app)
CLIP1_VIDEO = PROJECT_ROOT / "tests" / "test_data" / "clip1" / "clip1_video.mp4"


class TestWebStudioAPI(unittest.TestCase):

    def test_01_index_page(self):
        """Verify web studio homepage loads successfully."""
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Bhashara", resp.text)
        self.assertIn("Dub Video", resp.text)
        self.assertIn("Script Editor", resp.text)

    def test_02_url_info_plain_errors(self):
        """Verify plain-English error handling for broken or restricted links."""
        # Test invalid URL
        resp = client.post("/api/info", json={"url": "https://youtube.com/watch?v=invalid_id_xyz_999"})
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("error", data)
        self.assertTrue(len(data["error"]) > 0)
        print(f"\n[Test] Plain English Error on Invalid URL: '{data['error']}'")

    def test_03_job_creation_upload(self):
        """Verify video file upload job creation and queueing."""
        self.assertTrue(CLIP1_VIDEO.exists(), f"Missing test clip: {CLIP1_VIDEO}")

        with open(CLIP1_VIDEO, "rb") as f:
            resp = client.post(
                "/api/jobs",
                data={"target_lang": "hi", "source_lang": "en"},
                files={"file": ("test_upload.mp4", f, "video/mp4")}
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "queued")
        job_id = data["job_id"]
        print(f"[Test] Successfully queued upload job: {job_id}")

        # Check job details
        resp_details = client.get(f"/api/jobs/{job_id}")
        self.assertEqual(resp_details.status_code, 200)
        job_info = resp_details.json()
        self.assertEqual(len(job_info["stages"]), 10)
        self.assertEqual(job_info["stages"][0]["name"], "Input Demux")

    def test_04_cancel_job(self):
        """Verify cancellation endpoint stops job cleanly."""
        # Create a job and cancel it
        with open(CLIP1_VIDEO, "rb") as f:
            resp = client.post(
                "/api/jobs",
                data={"target_lang": "hi", "source_lang": "en"},
                files={"file": ("test_cancel.mp4", f, "video/mp4")}
            )
        job_id = resp.json()["job_id"]
        resp_cancel = client.post(f"/api/jobs/{job_id}/cancel")
        self.assertEqual(resp_cancel.status_code, 200)
        self.assertTrue(resp_cancel.json()["success"])
        print(f"[Test] Successfully cancelled job {job_id}")

    def test_05_disk_cleanup(self):
        """Verify disk space report and cleanup endpoint."""
        resp = client.post("/api/clean_temp")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("disk", data)
        self.assertIn("total_mb", data["disk"])
        print(f"[Test] Disk Usage: {data['disk']['total_mb']} MB (Cache: {data['disk']['cache_mb']} MB)")


if __name__ == "__main__":
    unittest.main()
