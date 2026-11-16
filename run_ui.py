"""
Bhashara Local Web Studio Launcher (run_ui.py)
Starts the local Bhashara web application on localhost (127.0.0.1)
and opens the interface in your default browser.
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def open_browser(url: str):
    """Waits for the server to bind and opens default browser."""
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 80)
    print(" 🎙️  BHASHARA LOCAL AI VIDEO DUBBING STUDIO")
    print("=" * 80)
    print(f"  Access URL:     {url}")
    print(f"  Environment:    Localhost Only (Private & Offline)")
    print(f"  Architecture:   FastAPI + Real Deep-Learning Inference")
    print("=" * 80)
    print("Opening web interface in your default browser...\n")

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(
        "ui.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False
    )


if __name__ == "__main__":
    main()
