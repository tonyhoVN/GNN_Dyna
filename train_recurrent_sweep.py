import copy
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "gnn_contact_direct_recurrent.json"
TRAIN_SCRIPT = REPO_ROOT / "train_recurrent.py"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def set_history_and_horizon(config: dict, history_len: int, pred_horizon: int) -> None:
    model_cfg = config.setdefault("model", {})
    node_encoder_cfg = model_cfg.setdefault("node_encoder", {})
    decoder_cfg = model_cfg.setdefault("decoder", {})

    node_encoder_cfg["history_len"] = int(history_len)
    decoder_cfg["pred_horizon"] = int(pred_horizon)


def run_training() -> None:
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", str(CONFIG_PATH)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    original_config = load_config(CONFIG_PATH)
    config = copy.deepcopy(original_config)

    sweeps = []
    sweeps.extend((h, 1) for h in range(1, 6))
    sweeps.extend((5, p) for p in range(2, 6))

    try:
        for history_len, pred_horizon in sweeps:
            print(f"\n=== Running history_len={history_len}, pred_horizon={pred_horizon} ===")
            set_history_and_horizon(config, history_len, pred_horizon)
            save_config(CONFIG_PATH, config)
            run_training()
    finally:
        save_config(CONFIG_PATH, original_config)
        print(f"\nRestored original config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
