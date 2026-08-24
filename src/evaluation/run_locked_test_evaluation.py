"""Run all locked-test graph, policy, error-profile, and figure outputs."""

import subprocess
from pathlib import Path

from src.evaluation.evaluate_locked_test_graph_policy import main as graph_policy_main
from src.evaluation.locked_test_error_analysis import main as error_analysis_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "reports" / "locked_test_reproducibility_manifest.txt"
RUN_COMMAND = ".\\.venv\\Scripts\\python.exe -m src.evaluation.run_locked_test_evaluation"
INPUT_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "rotation_dataset_2023.csv",
    PROJECT_ROOT / "data" / "processed" / "graph" / "tail_edges_2023.parquet",
    PROJECT_ROOT / "models" / "xgboost_propagation_2023_time_split.pkl",
]


def git_commit_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    for path in INPUT_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    graph_policy_main()
    error_analysis_main()
    manifest = (
        "LOCKED TEST REPRODUCIBILITY MANIFEST\n"
        "====================================\n"
        f"Command: {RUN_COMMAND}\n"
        f"Git commit hash: {git_commit_hash()}\n"
        "Frozen threshold: tau=0.46\n"
        "Evaluation period: 2023-11-01 through 2023-12-31 (inclusive)\n"
        "Input files:\n"
        + "\n".join(f"- {path}" for path in INPUT_PATHS)
        + "\n"
    )
    MANIFEST_PATH.write_text(manifest, encoding="utf-8")
    print(f"Saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
