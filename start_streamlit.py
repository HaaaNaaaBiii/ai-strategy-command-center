from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    out_log = (ROOT / "streamlit.out.log").open("w", encoding="utf-8")
    err_log = (ROOT / "streamlit.err.log").open("w", encoding="utf-8")
    creationflags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        creationflags |= subprocess.DETACHED_PROCESS
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    creationflags |= 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
    process = subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=out_log,
        stderr=err_log,
        creationflags=creationflags,
    )
    print(process.pid)


if __name__ == "__main__":
    main()
