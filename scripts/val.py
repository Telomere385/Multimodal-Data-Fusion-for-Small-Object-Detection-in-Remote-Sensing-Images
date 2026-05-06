import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a custom YOLOv11-RGBT model.")
    parser.add_argument("--config", help="Validation config YAML path.")
    parser.add_argument("--weights", help="Checkpoint path.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--split")
    parser.add_argument("--use-simotm", dest="use_simotm")
    parser.add_argument("--channels", type=int)
    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config


def merge_args(args):
    if not args.config:
        if not args.weights or not args.data:
            raise ValueError("--weights and --data are required when --config is not used.")
        return vars(args)

    config = load_config(args.config)
    merged = dict(config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        merged[key] = value

    if "weights" not in merged or "data" not in merged:
        raise ValueError("Config must define 'weights' and 'data', or provide them via CLI.")
    merged.setdefault("imgsz", 640)
    merged.setdefault("batch", 16)
    merged.setdefault("device", "0")
    merged.setdefault("project", "runs/val")
    merged.setdefault("name", "exp")
    merged.setdefault("split", "val")
    merged.setdefault("use_simotm", "RGBT")
    merged.setdefault("channels", 4)
    return merged


def main():
    args = parse_args()
    cfg = merge_args(args)
    model = YOLO(cfg["weights"])
    model.val(
        data=cfg["data"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["name"],
        split=cfg["split"],
        use_simotm=cfg["use_simotm"],
        channels=cfg["channels"],
    )


if __name__ == "__main__":
    main()
