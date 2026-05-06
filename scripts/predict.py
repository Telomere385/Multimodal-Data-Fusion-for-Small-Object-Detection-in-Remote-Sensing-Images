import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a custom YOLOv11-RGBT model.")
    parser.add_argument("--config", help="Prediction config YAML path.")
    parser.add_argument("--weights", help="Checkpoint path.")
    parser.add_argument("--source", help="Image, directory, video, or stream source.")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--use-simotm", dest="use_simotm")
    parser.add_argument("--channels", type=int)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-frames", dest="save_frames", action="store_true")
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
        if not args.weights or not args.source:
            raise ValueError("--weights and --source are required when --config is not used.")
        return vars(args)

    config = load_config(args.config)
    merged = dict(config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        if key in {"save", "save_frames"}:
            if value:
                merged[key] = True
            elif key not in merged:
                merged[key] = False
            continue
        merged[key] = value

    if "weights" not in merged or "source" not in merged:
        raise ValueError("Config must define 'weights' and 'source', or provide them via CLI.")
    merged.setdefault("imgsz", 640)
    merged.setdefault("device", "0")
    merged.setdefault("project", "runs/predict")
    merged.setdefault("name", "exp")
    merged.setdefault("conf", 0.25)
    merged.setdefault("use_simotm", "RGBT")
    merged.setdefault("channels", 4)
    merged.setdefault("save", False)
    merged.setdefault("save_frames", False)
    return merged


def main():
    args = parse_args()
    cfg = merge_args(args)
    model = YOLO(cfg["weights"])
    model.predict(
        source=cfg["source"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["name"],
        conf=cfg["conf"],
        use_simotm=cfg["use_simotm"],
        channels=cfg["channels"],
        save=cfg["save"],
        save_frames=cfg["save_frames"],
    )


if __name__ == "__main__":
    main()
