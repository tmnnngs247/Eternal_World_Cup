import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["build_dataset.py", "train_embeddings.py", "archetypes.py", "legend_score.py", "similarity.py"]

for script in SCRIPTS:
    print(f"\n=== Running {script} ===")
    subprocess.check_call([sys.executable, str(ROOT / "src" / script)], cwd=str(ROOT / "src"))
