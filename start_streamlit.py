from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    out_log = (ROOT / "streamlit.out.log").open("w", encoding="utf-8")
    err_log = (ROOT / "streamlit.err.log").open("w", encoding="utf-8")
    env = dict(os.environ)
    if "PATH" not in env and "Path" in env:
        env["PATH"] = env["Path"]
    env["PYTHONUNBUFFERED"] = "1"
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
        env=env,
    )
    print(process.pid)


if __name__ == "__main__":
    main()
