# YOLOv11-RGBT

这是一个基于 Ultralytics / YOLOv11 修改的多模态目标检测工程，支持 `RGBT`、`RGBRGB6C`、`Gray`、`SimOTM` 等数据读取与训练模式。

## 项目结构

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

仓库保留了修改后的 `ultralytics` 核心代码，并移除了文档、测试、示例资源和本地训练输出，更适合作为你自己的 GitHub 工程维护。

## 环境配置

```powershell
conda create -n yolov11-rgbt python=3.10 -y
conda activate yolov11-rgbt
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
pip install albumentations pycocotools
```

如果只使用 CPU：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install albumentations pycocotools
```

## 数据组织约定

在 `RGBT` 模式下，可见光和红外图像需要保持相同的相对路径，程序会自动把路径中的 `visible` 替换成 `infrared` 完成配对。

示例：

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

对应实现位于 `ultralytics/data/base.py` 和 `ultralytics/data/loaders.py`。

## 使用方式

现在推荐使用配置文件驱动训练、验证和推理。你只需要把实验参数写进 `configs/` 目录下的 YAML 文件。

训练配置示例：

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

使用配置文件训练：

```powershell
python scripts/train.py --config configs/train/example_rgbt.yaml
```

如果只想临时覆盖某几个参数，可以直接在命令行追加：

```powershell
python scripts/train.py --config configs/train/example_rgbt.yaml --device 1 --batch 8
```

如果不使用 `--config`，那么必须显式提供 `--model` 和 `--data`。

使用配置文件验证：

```powershell
python scripts/val.py --config configs/val/example_rgbt.yaml
```

同样支持临时覆盖：

```powershell
python scripts/val.py --config configs/val/example_rgbt.yaml --batch 8 --device 1
```

使用配置文件推理：

```powershell
python scripts/predict.py --config configs/predict/example_rgbt.yaml
```

同样支持临时覆盖：

```powershell
python scripts/predict.py --config configs/predict/example_rgbt.yaml --conf 0.4
```

## RGBT 四通道实现说明

默认 YOLO 只假设输入是 3 通道 RGB 图像，而这个仓库为了支持 `RGBT` 四通道输入，对源码做了成体系的修改。核心思路不是简单把第一层卷积从 3 通道改成 4 通道，而是把整条输入链路都改造成了“多模态输入系统”。

### 1. 参数层新增了通道数和模态类型

仓库在 [ultralytics/cfg/default.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/default.yaml:131) 中增加了两个关键参数：

- `channels`
- `use_simotm`

其中：

- `channels` 用来声明模型输入通道数，例如 `RGBT` 时通常设为 `4`
- `use_simotm` 用来声明输入模式，例如 `RGBT`、`RGBRGB6C`、`Gray`、`SimOTM`

这样训练、验证、推理都可以显式知道当前处理的是哪种模态输入，而不是默认按 3 通道 RGB 处理。

### 2. 数据加载层负责把 visible 和 infrared 拼成 4 通道

最关键的改动在 [ultralytics/data/base.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/base.py:267) 的 `load_image()`。

在 `RGBT` 分支中，代码会：

1. 读取 visible 图像，得到 3 通道 BGR
2. 将路径中的 `visible` 替换为 `infrared`，读取对应红外图像，得到 1 通道灰度图
3. 如果两个模态尺寸不同，分别 resize
4. 最后将 `B,G,R` 和红外通道合并成一个 4 通道图像

也就是说，进入网络前的数据不再是“两张图”，而是一张 `H x W x 4` 的四通道图像。

同样的逻辑在推理加载器 [ultralytics/data/loaders.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/loaders.py:547) 里也实现了一遍，因为推理阶段不走训练时的数据集类。

### 3. train / val / predict 全链路透传四通道信息

为了让数据构建、warmup、推理过程都知道当前输入不是默认的 3 通道，仓库把 `use_simotm` 和 `channels` 一路传了下去。

关键位置包括：

- [ultralytics/data/build.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/build.py:96)
- [ultralytics/models/yolo/detect/train.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/models/yolo/detect/train.py:43)
- [ultralytics/models/yolo/detect/val.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/models/yolo/detect/val.py:243)
- [ultralytics/engine/predictor.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/predictor.py:250)
- [ultralytics/engine/validator.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/validator.py:98)

比如验证阶段的 warmup 已经不是默认的 `(batch, 3, H, W)`，而是改成了 `(batch, channels, H, W)`，见 [ultralytics/engine/validator.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/validator.py:163)。

### 4. 数据增强和可视化也做了四通道适配

只修改读取层还不够，因为默认的数据增强和显示流程通常只服务于 1 通道或 3 通道图像。

仓库在 [ultralytics/data/augment.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/augment.py:1074) 中增加了按通道数分支处理的逻辑，例如：

- 4 通道 padding 使用 `(114, 114, 114, 114)`
- 区分 1 / 3 / 4 / 6 通道做不同处理
- 对 4 通道图像通常只对前 3 个 BGR 通道做颜色类增强，红外通道单独保留或单独处理

可视化和推理显示也做了适配，例如：

- [ultralytics/utils/plotting.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/utils/plotting.py:917)
- [ultralytics/engine/predictor.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/predictor.py:452)

因为 4 通道图像不能再按普通 RGB 图像直接显示，所以这里专门增加了对 4 通道和 6 通道的分支显示逻辑。

### 5. 模型结构层真正支持四通道输入和双分支融合

模型 YAML 中明确把输入通道改成了 `ch: 4`，例如：
[ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml:17)

但作者并没有把 4 通道直接塞进一个普通 backbone，而是通过自定义模块把 4 通道再次拆成两个分支：

- `SilenceChannel [0,3]`：取前 3 个通道，作为 visible 分支
- `SilenceChannel [3,4]`：取第 4 个通道，作为 infrared 分支

这个模块定义在：
[ultralytics/nn/modules/conv.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/modules/conv.py:352)

本质就是按通道切片：

```python
return x[..., self.c_start:self.c_end, :, :]
```

模型解析器也专门对这个模块做了支持，见：
[ultralytics/nn/tasks.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/tasks.py:1091)

解析器知道 `SilenceChannel[a,b]` 的输出通道数是 `b-a`，这样后续网络层的 shape 推导才能正确进行。

### 6. 以 midfusion 为例，网络如何处理 RGBT

以 [yolo11-RGBT-midfusion.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml:21) 为例：

1. 输入首先是 4 通道张量
2. 通过 `SilenceChannel [0,3]` 切出 visible 三通道
3. 通过 `SilenceChannel [3,4]` 切出 infrared 单通道
4. 两个分支分别走自己的 backbone
5. 在中间层通过 `Concat` 等方式融合 visible 和 infrared 特征
6. 融合后的特征再进入检测头

所以这个仓库的核心不是“把第一层卷积改成 4 通道”，而是“4 通道输入 + 双分支建模 + 指定层融合”。

### 7. 和默认 YOLO 的本质区别

默认 YOLO：

- 假设输入就是单张 3 通道 RGB 图像
- 数据加载只读一张图
- 数据增强默认只考虑 1 通道或 3 通道
- 模型 stem 默认只服务于单路输入

这个仓库的 RGBT 版本：

- 数据层支持 visible / infrared 自动配对
- 输入层支持拼成 4 通道张量
- train / val / predict 全链路支持 `channels=4`
- 数据增强和显示流程支持 4 通道
- 模型结构支持 visible / infrared 双分支
- 通过 early / mid / late fusion 等方式进行多模态融合

### 8. 最值得优先阅读的源码位置

如果只想快速理解四通道 RGBT 是怎么做出来的，建议优先看下面这些文件：

- [ultralytics/data/base.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/base.py:267)
- [ultralytics/data/loaders.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/loaders.py:547)
- [ultralytics/nn/modules/conv.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/modules/conv.py:352)
- [ultralytics/nn/tasks.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/nn/tasks.py:1091)
- [ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml](E:/master/github_project/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml:17)
- [ultralytics/data/augment.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/data/augment.py:1074)
- [ultralytics/engine/predictor.py](E:/master/github_project/YOLOv11-RGBT/ultralytics/engine/predictor.py:117)

一句话总结：这个仓库对源码的深度修改，不在于“把第一层卷积改成 4 输入通道”，而在于把 Ultralytics 的整条输入链路都改造成了多模态输入系统。

## 说明

- 自定义融合模型配置位于 `ultralytics/cfg/models/*-RGBT/`。
- `ultralytics/cfg/datasets/` 下的数据集 YAML 仍保留了原作者本地路径，需要按你自己的数据位置修改。
- 训练配置示例见 [configs/train/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/train/example_rgbt.yaml:1)。
- 验证配置示例见 [configs/val/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/val/example_rgbt.yaml:1)。
- 推理配置示例见 [configs/predict/example_rgbt.yaml](E:/master/github_project/YOLOv11-RGBT/configs/predict/example_rgbt.yaml:1)。
- 大权重文件、数据集和 `runs/` 输出默认不纳入 Git 管理。
