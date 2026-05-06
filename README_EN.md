# YOLOv11-RGBT

Custom YOLOv11/Ultralytics fork for grayscale and multispectral object detection, including `RGBT`, `RGBRGB6C`, `Gray`, and `SimOTM` data-loading modes.

## Project Layout

```text
.
|-- configs/
|   |-- predict/
|   |   `-- example_rgbt.yaml
|   |-- train/
|   |   `-- example_rgbt.yaml
|   `-- val/
|       `-- example_rgbt.yaml
|-- pyproject.toml
|-- README.md
|-- README_Zh.md
|-- scripts/
|   |-- train.py
|   |-- val.py
|   `-- predict.py
`-- ultralytics/
```

The repository keeps the modified `ultralytics` package and removes documentation, tests, example assets, and local training outputs so it can be published as a focused project.

## Environment

```powershell
conda create -n yolov11-rgbt python=3.10 -y
conda activate yolov11-rgbt
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
pip install albumentations pycocotools
```

For CPU-only usage:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install albumentations pycocotools
```

## Dataset Convention

For `RGBT` mode, the visible and infrared files must share the same relative path, with `visible` replaced by `infrared`.

Example:

```text
dataset/
|-- images/
|   |-- visible/
|   |   |-- train/
|   |   `-- val/
|   `-- infrared/
|       |-- train/
|       `-- val/
`-- labels/
    |-- visible/
    |   |-- train/
    |   `-- val/
    `-- infrared/
        |-- train/
        `-- val/
```

The custom pairing logic is implemented in `ultralytics/data/base.py` and `ultralytics/data/loaders.py`.

## Usage

Training is now config-first. Put your experiment settings in a YAML file under `configs/train/`.

Example config:

```yaml
model: ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml
data: path/to/your_dataset.yaml

epochs: 100
imgsz: 640
batch: 16
workers: 4
device: "0"
optimizer: SGD

project: runs/train
name: example_rgbt

use_simotm: RGBT
channels: 4
close_mosaic: 0
cache: false
```

Train with a config file:

```powershell
python scripts/train.py --config configs/train/example_rgbt.yaml
```

You can still override individual fields from the command line when needed:

```powershell
python scripts/train.py --config configs/train/example_rgbt.yaml --device 1 --batch 8
```

If you do not use `--config`, then `--model` and `--data` must be provided explicitly.

Validate with a config file:

```powershell
python scripts/val.py --config configs/val/example_rgbt.yaml
```

You can override fields when needed:

```powershell
python scripts/val.py --config configs/val/example_rgbt.yaml --batch 8 --device 1
```

Predict with a config file:

```powershell
python scripts/predict.py --config configs/predict/example_rgbt.yaml
```

You can override fields when needed:

```powershell
python scripts/predict.py --config configs/predict/example_rgbt.yaml --conf 0.4
```

## Notes

- The custom fusion models are under `ultralytics/cfg/models/*-RGBT/`.
- Dataset YAML files under `ultralytics/cfg/datasets/` still contain the original author paths and should be edited for your local datasets.
- The example training config is [configs/train/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/train/example_rgbt.yaml:1).
- The example validation config is [configs/val/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/val/example_rgbt.yaml:1).
- The example prediction config is [configs/predict/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/predict/example_rgbt.yaml:1).
- Large weights, datasets, and `runs/` outputs are intentionally excluded from version control.
