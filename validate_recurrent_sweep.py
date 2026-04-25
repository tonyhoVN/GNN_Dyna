import copy
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "benchmark" / "model_validate.json"
VALIDATE_SCRIPT = REPO_ROOT / "validate_model.py"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def resolve_path(path_str: str | None, default: Path | None = None) -> Path | None:
    if not path_str:
        return default
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def set_history_and_horizon(config: dict, history_len: int, pred_horizon: int) -> None:
    model_cfg = config.setdefault("model", {})
    node_encoder_cfg = model_cfg.setdefault("node_encoder", {})
    decoder_cfg = model_cfg.setdefault("decoder", {})
    node_encoder_cfg["history_len"] = int(history_len)
    decoder_cfg["pred_horizon"] = int(pred_horizon)


def run_validation(config_path: Path, pt_path: Path, start_index: int, rollout_steps: int, extra_args: list[str]) -> None:
    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--config",
        str(config_path),
        "--pt-file",
        str(pt_path),
        "--start-index",
        str(int(start_index)),
        "--rollout-steps",
        str(int(rollout_steps)),
        *extra_args,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    manifest = load_json(MANIFEST_PATH)
    pt_files = manifest.get("pt_files", [])
    if not pt_files:
        raise ValueError("benchmark/model_validate.json must contain a non-empty 'pt_files' list.")

    sweeps = []
    sweeps.extend((h, 1) for h in range(1, 6))
    sweeps.extend((5, p) for p in range(2, 6))
    if len(pt_files) != len(sweeps):
        raise ValueError(
            f"pt_files has {len(pt_files)} entries but the sweep has {len(sweeps)} runs. "
            "Make them the same length."
        )

    default_config_path = resolve_path(manifest.get("config"))
    if default_config_path is None:
        default_config_path = REPO_ROOT / "config" / "gnn_contact_direct_recurrent_infer.json"

    start_index = int(manifest.get("test_data", {}).get("start_index", 20))
    rollout_steps = int(manifest.get("test_data", {}).get("rollout_steps", 150))

    original_config = load_json(default_config_path)
    config = copy.deepcopy(original_config)

    try:
        for idx, ((history_len, pred_horizon), pt_file) in enumerate(zip(sweeps, pt_files), start=1):
            pt_path = resolve_path(pt_file)
            if pt_path is None:
                raise ValueError(f"Run {idx} is missing a checkpoint path.")

            print(
                f"\n=== Validation {idx}/{len(sweeps)}: "
                f"history_len={history_len}, pred_horizon={pred_horizon} ==="
            )
            set_history_and_horizon(config, history_len, pred_horizon)
            save_json(default_config_path, config)
            run_validation(
                config_path=default_config_path,
                pt_path=pt_path,
                start_index=start_index,
                rollout_steps=rollout_steps,
                extra_args=[],
            )
    finally:
        save_json(default_config_path, original_config)
        print(f"\nRestored original config: {default_config_path}")


if __name__ == "__main__":
    main()
