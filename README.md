# YOLOv11-RGBT

This is a multimodal object detection project modified from Ultralytics / YOLOv11. It supports multiple data loading and training modes, including `RGBT`, `RGBRGB6C`, `Gray`, and `SimOTM`.

## Project Structure

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

The repository retains the modified core `ultralytics` source code while removing documentation, tests, example resources, and local training outputs, making it more suitable for maintaining as a standalone GitHub project.

## Environment Setup

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

## Dataset Organization

In `RGBT` mode, visible-light and infrared images must follow the same relative directory structure. The program automatically replaces `visible` with `infrared` in the image path to locate the corresponding infrared image.

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

The corresponding implementation can be found in `ultralytics/data/base.py` and `ultralytics/data/loaders.py`.

## Usage

The recommended workflow is now configuration-file-driven training, validation, and inference. Experiment parameters only need to be specified in YAML files under the `configs/` directory.

Example training configuration:

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

Train using a configuration file:

```powershell
python scripts/train.py --config configs/train/example_rgbt.yaml
```

To temporarily override selected parameters, append them directly to the command:

```powershell
python scripts/train.py --config configs/train/example_rgbt.yaml --device 1 --batch 8
```

If `--config` is not used, both `--model` and `--data` must be explicitly provided.

Validate using a configuration file:

```powershell
python scripts/val.py --config configs/val/example_rgbt.yaml
```

Temporary parameter overrides are also supported:

```powershell
python scripts/val.py --config configs/val/example_rgbt.yaml --batch 8 --device 1
```

Run inference using a configuration file:

```powershell
python scripts/predict.py --config configs/predict/example_rgbt.yaml
```

Temporary parameter overrides are also supported:

```powershell
python scripts/predict.py --config configs/predict/example_rgbt.yaml --conf 0.4
```

## RGBT Four-Channel Implementation

Standard YOLO assumes a 3-channel RGB input. To support 4-channel `RGBT` input, this repository introduces systematic modifications throughout the source code.

The core idea is not simply to change the first convolutional layer from 3 input channels to 4. Instead, the entire input pipeline is extended into a multimodal input system.

### 1. Channel Count and Modality Type Parameters

Two key parameters were added to [ultralytics/cfg/default.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/default.yaml:131):

* `channels`
* `use_simotm`

Specifically:

* `channels` specifies the number of input channels. For `RGBT`, it is typically set to `4`.
* `use_simotm` specifies the input modality, such as `RGBT`, `RGBRGB6C`, `Gray`, or `SimOTM`.

This allows the training, validation, and inference pipelines to explicitly determine the current input modality instead of assuming standard 3-channel RGB input.

### 2. The Data Loader Combines Visible and Infrared Images into Four Channels

The most important modification is located in `load_image()` in [ultralytics/data/base.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/base.py:267).

In the `RGBT` branch, the code:

1. Loads the visible image as a 3-channel BGR image.
2. Replaces `visible` with `infrared` in the file path and loads the corresponding infrared image as a single-channel grayscale image.
3. Resizes the two modalities independently if their spatial dimensions differ.
4. Concatenates the `B,G,R` channels with the infrared channel to form a four-channel image.

Therefore, the input passed to the network is no longer represented as two separate images, but as a single tensor with shape `H x W x 4`.

The same logic is implemented separately in the inference loader [ultralytics/data/loaders.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/loaders.py:547), because inference does not use the same dataset class as training.

### 3. Four-Channel Information Is Propagated Through Training, Validation, and Inference

To ensure that dataset construction, model warmup, and inference correctly handle non-standard input channels, `use_simotm` and `channels` are propagated throughout the entire pipeline.

Key locations include:

* [ultralytics/data/build.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/build.py:96)
* [ultralytics/models/yolo/detect/train.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/models/yolo/detect/train.py:43)
* [ultralytics/models/yolo/detect/val.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/models/yolo/detect/val.py:243)
* [ultralytics/engine/predictor.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/predictor.py:250)
* [ultralytics/engine/validator.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/validator.py:98)

For example, the validation warmup tensor is no longer hard-coded as `(batch, 3, H, W)`. Instead, it uses `(batch, channels, H, W)`, as implemented in [ultralytics/engine/validator.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/validator.py:163).

### 4. Data Augmentation and Visualization Support Four-Channel Input

Modifying only the image-loading pipeline is not sufficient because standard data augmentation and visualization pipelines are generally designed for one-channel or three-channel images.

Additional channel-dependent processing logic was introduced in [ultralytics/data/augment.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/augment.py:1074), including:

* Four-channel padding using `(114, 114, 114, 114)`.
* Different processing branches for 1-, 3-, 4-, and 6-channel inputs.
* Color-based augmentation is generally applied only to the first three BGR channels, while the infrared channel is preserved or processed separately.

Visualization and inference display logic were also adapted, including:

* [ultralytics/utils/plotting.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/utils/plotting.py:917)
* [ultralytics/engine/predictor.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/predictor.py:452)

Because four-channel images cannot be displayed directly as standard RGB images, separate visualization branches were introduced for four-channel and six-channel inputs.

### 5. The Model Architecture Supports Four-Channel Input and Dual-Branch Fusion

The model YAML explicitly sets the input channel count to `ch: 4`, for example:

[ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml:17)

However, instead of directly feeding the four channels into a conventional backbone, the input is split back into two branches using custom modules:

* `SilenceChannel [0,3]`: extracts the first three channels for the visible branch.
* `SilenceChannel [3,4]`: extracts the fourth channel for the infrared branch.

The module is defined in:

[ultralytics/nn/modules/conv.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/modules/conv.py:352)

Conceptually, the implementation is simply channel slicing:

```python
return x[..., self.c_start:self.c_end, :, :]
```

The model parser was also extended to support this module:

[ultralytics/nn/tasks.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/tasks.py:1091)

The parser determines that the number of output channels from `SilenceChannel[a,b]` is `b-a`, allowing subsequent network layers to infer tensor dimensions correctly.

### 6. How RGBT Is Processed in the Mid-Fusion Architecture

Using [yolo11-RGBT-midfusion.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml:21) as an example:

1. The model first receives a four-channel input tensor.
2. `SilenceChannel [0,3]` extracts the three visible channels.
3. `SilenceChannel [3,4]` extracts the single infrared channel.
4. The two modalities are processed by separate backbone branches.
5. Visible and infrared features are fused at intermediate layers using operations such as `Concat`.
6. The fused features are then passed to the detection head.

Therefore, the core design of this repository is not simply “changing the first convolution to accept four channels,” but rather:

**four-channel input + dual-branch feature extraction + fusion at selected network layers.**

### 7. Key Differences from Standard YOLO

Standard YOLO:

* Assumes a single 3-channel RGB image as input.
* Loads only one image per sample.
* Data augmentation primarily assumes one-channel or three-channel images.
* The model stem is designed for a single input stream.

This RGBT implementation:

* Automatically pairs visible and infrared images at the data-loading stage.
* Combines them into a four-channel input tensor.
* Supports `channels=4` throughout training, validation, and inference.
* Adapts data augmentation and visualization for four-channel inputs.
* Supports separate visible and infrared backbone branches.
* Supports multimodal fusion strategies such as early, mid, and late fusion.

### 8. Recommended Source Files to Read First

To quickly understand how four-channel RGBT processing is implemented, the following files are the most important:

* [ultralytics/data/base.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/base.py:267)
* [ultralytics/data/loaders.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/loaders.py:547)
* [ultralytics/nn/modules/conv.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/modules/conv.py:352)
* [ultralytics/nn/tasks.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/tasks.py:1091)
* [ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml:17)
* [ultralytics/data/augment.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/augment.py:1074)
* [ultralytics/engine/predictor.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/predictor.py:117)

In short, the main modification in this repository is not simply changing the first layer to accept four input channels. Instead, the entire Ultralytics input pipeline has been extended into a multimodal input system.

## Added Model Capabilities

### YOLOv11_RGBT + Residual Fusion

The repository introduces a `YOLOv11_RGBT + Residual Fusion` architecture as an alternative to the direct RGB/T feature `Concat` used in the original mid-fusion model.

The fusion module uses RGB features as the main path and injects infrared features as a learnable residual component:

```python
fused = rgb_proj + alpha * thermal_proj
```

Key updates:

* Added the `RGBTResidualFusion` module.
* RGB and thermal features are first aligned to the same number of channels using `1x1 Conv`.
* `alpha` is a learnable parameter initialized to `0.1`, preventing the thermal branch from excessively disturbing the RGB main path during the early stages of training.
* The RGB/T fusion points at P3, P4, and P5 are all replaced with residual fusion.
* Added the configuration file: [ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-Residual.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-Residual.yaml:1)

To train this model, change the `model` entry in the configuration to:

```yaml
model: ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-Residual.yaml
use_simotm: RGBT
channels: 4
```

This version is recommended as a primary experimental configuration. The original `Concat` mid-fusion model, Residual Fusion, and BiFPN can be used as three ablation variants.

### YOLOv11_RGBT + BiFPN

The repository also introduces a `YOLOv11_RGBT + BiFPN` architecture, which adds a learnable weighted BiFPN neck after dual-branch RGBT feature fusion.

Key updates:

* Added a `BiFPNFusion` module that supports channel alignment, spatial-resolution alignment, and learnable normalized weighted fusion across multiple input features.
* `parse_model()` has been extended so that `BiFPNFusion` can be directly specified in model YAML files.
* Added the configuration file: [ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-BiFPN.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-BiFPN.yaml:1)

To train this model, change the `model` entry in the configuration to:

```yaml
model: ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-BiFPN.yaml
use_simotm: RGBT
channels: 4
```

### YOLOv11_RGBT + SR Auxiliary Branch

The repository introduces a `YOLOv11_RGBT + SR` super-resolution auxiliary branch.

The architecture keeps the original detection head unchanged while adding an auxiliary reconstruction branch during training. The branch reconstructs the four-channel RGBT input from shallow fused features, and the SR reconstruction loss is added to the overall training objective to improve multimodal feature representation.

Key updates:

* Added `SRHead`, which reconstructs the RGBT image by upsampling features from the P3/8 level.
* Added `DetectSR`, allowing detection outputs and SR outputs to coexist. Inference remains focused on object detection outputs.
* `v8DetectionLoss` has been extended to include `sr_loss`, using `L1 loss` by default.
* Added the `sr` hyperparameter to control the weight of the SR auxiliary loss, with a default value of `0.1`.
* Training logs for SR models report `box_loss / cls_loss / dfl_loss / sr_loss`.
* Added the configuration file: [ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-SR.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-SR.yaml:1)

To train this model, change the configuration to:

```yaml
model: ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-SR.yaml
use_simotm: RGBT
channels: 4
sr: 0.1
```

### Validation Status

The following static checks have been completed:

* Python syntax compilation checks passed.
* `yolo11-RGBT-midfusion-BiFPN.yaml` and `yolo11-RGBT-midfusion-SR.yaml` can be parsed successfully as YAML.
* Layer-reference indices in the SR configuration have been verified.

Full model instantiation and forward-pass validation require the current Python environment to import `ultralytics` successfully.

If compatibility problems occur between `NumPy 2.x` and older binary extensions from packages such as `matplotlib`, adjust the environment first, for example by downgrading to `numpy<2` or upgrading the relevant binary packages.

## Notes

* Custom fusion model configurations are located under `ultralytics/cfg/models/*-RGBT/`.
* Dataset YAML files under `ultralytics/cfg/datasets/` still contain local paths from the original author and should be modified according to your own dataset location.
* Example training configuration: [configs/train/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/train/example_rgbt.yaml:1).
* Example validation configuration: [configs/val/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/val/example_rgbt.yaml:1).
* Example inference configuration: [configs/predict/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/predict/example_rgbt.yaml:1).
* Large model weights, datasets, and `runs/` outputs are excluded from Git version control by default.
