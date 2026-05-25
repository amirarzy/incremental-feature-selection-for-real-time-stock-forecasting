from pathlib import Path
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"
LOG_DIR = RESULTS_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "Models/Baselines/M1_P.py",
    "Models/Baselines/M1_D.py",
    "Models/Baselines/M2_P.py",
    "Models/Baselines/M2_D.py",
    "Models/Baselines/M3_P.py",
    "Models/Baselines/M3_D.py",
    "Models/Baselines/M4_P.py",
    "Models/Baselines/M4_D.py",
    "Models/Baselines/M5_P.py",
    "Models/Baselines/M5_D.py",
    "Models/Baselines/M6_P.py",
    "Models/Baselines/M6_D.py",
    "Models/IFS_Models/M7_P.py",
    "Models/IFS_Models/M7_D.py",
]

def run_script(script: str) -> int:
    script_path = PROJECT_ROOT / script
    model_name = script_path.stem
    log_file = LOG_DIR / f"{model_name}.log"

    print("=" * 100)
    print(f"Running: {script}")
    print(f"Start:   {datetime.now()}")
    print(f"Log:     {log_file}")

    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}")
        return 1

    with open(log_file, "w", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    print(f"Finished: {script}")
    print(f"Return code: {process.returncode}")
    print(f"End:      {datetime.now()}")

    return process.returncode

def main():
    failed = []

    print(f"PROJECT ROOT: {PROJECT_ROOT}")
    print(f"RESULTS DIR:  {RESULTS_DIR}")
    print(f"LOG DIR:      {LOG_DIR}")

    for script in SCRIPTS:
        code = run_script(script)
        if code != 0:
            failed.append(script)
            print(f"FAILED: {script}")
            break

    print("=" * 100)
    if failed:
        print("RUN STOPPED. Failed script:")
        for f in failed:
            print(f"- {f}")
        sys.exit(1)

    print("ALL SCRIPTS FINISHED SUCCESSFULLY.")

if __name__ == "__main__":
    main()