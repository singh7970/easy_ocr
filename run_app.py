"""
run_app.py — Entry point for the packaged Windows EXE.

How it works:
  1. Resolves the correct path to bundled templates/ and static/
     (works both as a plain .py and inside a PyInstaller .exe)
  2. Injects APP_BASE_DIR so api.py can find templates/static from
     the PyInstaller temp folder (sys._MEIPASS) at runtime
  3. Ensures static/uploads/ exists next to the EXE (for saving images)
  4. Starts FastAPI (uvicorn) in a background thread
  5. Polls /health until the server is ready, then opens the browser
  6. Keeps the process alive until uvicorn exits

Usage (development):
    python run_app.py

Usage (build EXE on Windows):
    build_exe.bat
"""

import os
import sys
import time
import threading
import webbrowser
import urllib.request

# ── 1. Locate the bundle root ────────────────────────────────────────────────
# PyInstaller frozen EXE  → sys._MEIPASS (temp folder with bundled files)
# Plain .py dev run       → directory that contains this script
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    # EXE_DIR is where the .exe lives — keep user data (db, uploads) here
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR  = BASE_DIR

# ── 2. Expose paths for api.py ───────────────────────────────────────────────
# api.py reads APP_BASE_DIR to locate templates/ and static/
os.environ["APP_BASE_DIR"] = BASE_DIR
# api.py reads APP_DATA_DIR to know where to write uploads & the SQLite DB
os.environ["APP_DATA_DIR"] = EXE_DIR

# Make sure Python can find the project modules inside the bundle
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 3. Create runtime data folders next to the EXE ───────────────────────────
os.makedirs(os.path.join(EXE_DIR, "static", "uploads"), exist_ok=True)

# ── 4. Point EasyOCR at bundled models (offline support) ─────────────────────
# When built with --add-data "%USERPROFILE%\.EasyOCR\model;.EasyOCR\model",
# the models land inside sys._MEIPASS\.EasyOCR\model.
# Setting EASYOCR_MODULE_PATH tells EasyOCR to look there instead of downloading.
_bundled_models = os.path.join(BASE_DIR, ".EasyOCR", "model")
if os.path.isdir(_bundled_models):
    os.environ["EASYOCR_MODULE_PATH"] = os.path.join(BASE_DIR, ".EasyOCR")
    print(f"[run_app] EasyOCR models found in bundle: {_bundled_models}")

HOST = "127.0.0.1"
PORT = 8000
URL  = f"http://{HOST}:{PORT}"


# ── 4. Server thread ─────────────────────────────────────────────────────────
def run_server() -> None:
    import uvicorn
    from api import app as fastapi_app  # Your actual FastAPI app is in api.py
    uvicorn.run(fastapi_app, host=HOST, port=PORT, log_level="warning")


# ── 5. Poll until server is ready ────────────────────────────────────────────
def wait_for_server(url: str, timeout: int = 90) -> bool:
    """
    Poll / (root) until the server responds or timeout is reached.
    EasyOCR model loading can take 30-60s on first run, so we allow 90s.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── 6. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[run_app] Bundle dir : {BASE_DIR}")
    print(f"[run_app] Data dir   : {EXE_DIR}")
    print(f"[run_app] Starting server at {URL} …")
    print(f"[run_app] NOTE: First launch loads EasyOCR — may take 30-60 s")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    if wait_for_server(URL, timeout=90):
        print("[run_app] Server ready — opening browser.")
    else:
        print("[run_app] Timeout — opening browser anyway (server may still be loading).")

    webbrowser.open(URL)

    # Keep the process alive — the daemon server thread dies when this exits
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("[run_app] Shutting down.")
