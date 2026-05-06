import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom YOLOv11-RGBT model.")
    parser.add_argument("--config", help="Training config YAML path.")
    parser.add_argument("--model", help="Model YAML or pretrained weight path.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--optimizer")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--use-simotm", dest="use_simotm")
    parser.add_argument("--channels", type=int)
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int)
    parser.add_argument("--cache", action="store_true")
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
        if not args.model or not args.data:
            raise ValueError("--model and --data are required when --config is not used.")
        return vars(args)

    config = load_config(args.config)
    merged = dict(config)

    # CLI values override config values when explicitly provided.
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        if key == "cache":
            if value:
                merged[key] = True
            elif key not in merged:
                merged[key] = False
            continue
        merged[key] = value

    if "model" not in merged or "data" not in merged:
        raise ValueError("Config must define 'model' and 'data', or provide them via CLI.")
    merged.setdefault("epochs", 100)
    merged.setdefault("imgsz", 640)
    merged.setdefault("batch", 16)
    merged.setdefault("workers", 4)
    merged.setdefault("device", "0")
    merged.setdefault("optimizer", "SGD")
    merged.setdefault("project", "runs/train")
    merged.setdefault("name", "exp")
    merged.setdefault("use_simotm", "RGBT")
    merged.setdefault("channels", 4)
    merged.setdefault("close_mosaic", 0)
    merged.setdefault("cache", False)
    return merged


def main():
    args = parse_args()
    cfg = merge_args(args)
    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        device=cfg["device"],
        optimizer=cfg["optimizer"],
        project=cfg["project"],
        name=cfg["name"],
        use_simotm=cfg["use_simotm"],
        channels=cfg["channels"],
        close_mosaic=cfg["close_mosaic"],
        cache=cfg["cache"],
    )


if __name__ == "__main__":
    main()
