# YOLOv11-RGBT 答辩内容实现计划

## 1. 目标

根据答辩材料，需要在当前仓库基础上完成一条清晰、可复现实验主线：

1. 建立 `YOLOv11_RGBT` 基线模型
2. 实现 `YOLOv11_RGBT + BiFPN`
3. 实现 `YOLOv11_RGBT + A2C2f`
4. 实现 `YOLOv11_RGBT + SR`
5. 实现 `YOLOv11_RGBT + ALL`
6. 完成对应训练、验证、消融实验和可复现配置

最终目标不是继续扩展现有零散变体，而是让仓库能够完整对应答辩中的技术路线和实验结论。

---

## 2. 当前仓库现状

### 2.1 已有能力

当前仓库已经具备：

- RGBT 四通道输入支持
- `visible -> infrared` 自动配对读取
- early / mid / late / score fusion 等多模态融合框架
- YOLOv11-RGBT 中期融合基线雏形
- `A2C2f` 模块定义与解析支持
- 训练 / 验证 / 推理的 `--config` 配置化入口

### 2.2 当前缺失

答辩主线对应的关键缺口：

- 没有明确命名和固化的 `YOLOv11_RGBT baseline`
- 没有 `YOLOv11_RGBT + BiFPN` 的完整模块实现和模型 YAML
- 没有 `YOLOv11_RGBT + SR` 的超分辨率辅助分支实现
- 没有 `YOLOv11_RGBT + ALL` 的组合模型
- 没有围绕答辩主线整理出的标准实验配置
- 现有 `CAS/CMA/PGI/DBB` 等分支较多，但不属于答辩主线

---

## 3. 建议实施路线

建议按从低风险到高风险、从低耦合到高耦合的顺序推进：

1. 固化基线
2. 接入 A2C2f
3. 接入 BiFPN
4. 接入 SR 分支
5. 组合 ALL
6. 补齐训练配置、实验记录和文档

原因：

- `A2C2f` 现成度最高，最容易先落地
- `BiFPN` 需要新增结构模块，但不一定先改 loss
- `SR` 最复杂，因为除了结构还可能涉及辅助损失和训练逻辑
- `ALL` 必须建立在前三项都稳定的基础上

---

## 4. 分阶段计划

## 阶段一：固化答辩基线

### 4.1 目标

建立一个明确的 `YOLOv11_RGBT` 基线，作为所有后续改进的统一起点。

### 4.2 做什么

- 从当前 `yolo11-RGBT-midfusion.yaml` 复制出论文基线版本
- 统一命名、统一配置、统一训练入口
- 明确 baseline 的实验参数和数据集配置

### 4.3 怎么做

新增：

- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-baseline.yaml`

建议做法：

- 初始版本直接复制 `yolo11-RGBT-midfusion.yaml`
- 保留 `ch: 4`
- 保留 visible / infrared 双分支
- 保留中期特征融合路径

新增训练配置：

- `configs/train/baseline_rgbt.yaml`
- `configs/val/baseline_rgbt.yaml`
- `configs/predict/baseline_rgbt.yaml`

### 4.4 验收标准

- baseline 可以正常 train / val / predict
- 配置文件可以直接复现
- baseline 的实验结果可作为后续对照组

---

## 阶段二：实现 `YOLOv11_RGBT + A2C2f`

### 5.1 目标

将答辩中的 `A2C2f` 改进真正接入 YOLOv11-RGBT 主线。

### 5.2 当前状态

当前仓库已有：

- `A2C2f` 模块定义
- 模块注册与解析支持

但尚未形成答辩主线下的 `YOLOv11_RGBT + A2C2f` 模型。

### 5.3 做什么

- 在 baseline 基础上，替换 backbone 或 neck 中的部分 `C3k2 / C2f` 为 `A2C2f`
- 构建独立模型 YAML
- 建立独立训练配置

### 5.4 怎么做

重点文件：

- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-a2c2f.yaml`

参考来源：

- `ultralytics/nn/modules/block.py`
- `ultralytics/cfg/models/12/yolo12.yaml`

建议策略：

- 先只替换 neck 中关键层，降低训练不稳定风险
- 后续再尝试扩大到 backbone

新增配置：

- `configs/train/a2c2f_rgbt.yaml`
- `configs/val/a2c2f_rgbt.yaml`

### 5.5 验收标准

- 新模型可正常解析和训练
- 与 baseline 相比指标有可量化对比
- 日志和配置可复现

---

## 阶段三：实现 `YOLOv11_RGBT + BiFPN`

### 6.1 目标

完成答辩中的“双向特征金字塔”改进。

### 6.2 当前状态

当前仓库没有完整的 `BiFPN / BiConcat` 实现，也没有对应的 `YOLOv11_RGBT + BiFPN` 模型 YAML。

### 6.3 做什么

- 新增 `BiFPN` 相关模块
- 在模块导出和 YAML 解析中注册
- 将 baseline 中原有 neck 替换为 BiFPN 结构

### 6.4 怎么做

建议新增模块文件位置：

- 优先放在 `ultralytics/nn/modules/block.py`

需要改动的文件：

- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`

新增模型配置：

- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-bifpn.yaml`

实现要点：

- 双向跨尺度连接
- 可学习融合权重
- 模块化 repeat block
- 与现有 Detect head 对接

### 6.5 验收标准

- BiFPN 结构可被 YAML 正常解析
- 模型前向和训练正常
- 与 baseline 对比完成消融

---

## 阶段四：实现 `YOLOv11_RGBT + SR`

### 7.1 目标

实现答辩中的超分辨率辅助分支，用于增强小目标细节表征。

### 7.2 当前状态

当前仓库没有完整的 SR 分支、没有 EDSR 风格模块、也没有配套的辅助损失。

### 7.3 做什么

- 设计 SR 辅助分支
- 将浅层和高层特征结合后送入 SR 分支
- 将 SR 分支输出以辅助特征或辅助损失形式注入主干

### 7.4 怎么做

建议新增内容：

- SR 分支模块
- 如果采用重建监督，则增加 SR loss

可能需要修改的文件：

- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/models/yolo/detect/train.py`
- `ultralytics/engine/trainer.py`
- `ultralytics/utils/loss.py`

新增模型配置：

- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-sr.yaml`

新增训练配置：

- `configs/train/sr_rgbt.yaml`
- `configs/val/sr_rgbt.yaml`

### 7.5 风险

这是当前最复杂的一项，因为可能需要：

- 额外 loss
- 多分支训练
- 调整训练稳定性

### 7.6 验收标准

- SR 分支可被训练图正常调用
- 主干训练不报错
- 可以单独比较 `+SR` 和 baseline

---

## 阶段五：实现 `YOLOv11_RGBT + ALL`

### 8.1 目标

在统一基线上组合三项改进：

- BiFPN
- A2C2f
- SR

### 8.2 做什么

- 建立组合模型 YAML
- 统一训练配置
- 跑出最终实验结果

### 8.3 怎么做

新增：

- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-all.yaml`
- `configs/train/all_rgbt.yaml`
- `configs/val/all_rgbt.yaml`

组合原则：

- backbone / neck 的基础框架来自 baseline
- neck 融入 BiFPN
- 指定层替换为 A2C2f
- 挂接 SR 分支

### 8.4 验收标准

- `ALL` 模型完整可训练
- 与 `baseline / +BiFPN / +A2C2f / +SR` 完成统一对比
- 指标和答辩结果方向一致

---

## 5. 实验组织建议

建议围绕答辩主线建立如下实验配置：

### 9.1 训练配置

- `configs/train/baseline_rgbt.yaml`
- `configs/train/bifpn_rgbt.yaml`
- `configs/train/a2c2f_rgbt.yaml`
- `configs/train/sr_rgbt.yaml`
- `configs/train/all_rgbt.yaml`

### 9.2 验证配置

- `configs/val/baseline_rgbt.yaml`
- `configs/val/bifpn_rgbt.yaml`
- `configs/val/a2c2f_rgbt.yaml`
- `configs/val/sr_rgbt.yaml`
- `configs/val/all_rgbt.yaml`

### 9.3 结果管理建议

建议统一输出目录命名，例如：

- `runs/train/baseline_rgbt`
- `runs/train/bifpn_rgbt`
- `runs/train/a2c2f_rgbt`
- `runs/train/sr_rgbt`
- `runs/train/all_rgbt`

这样后续对比更清晰。

---

## 6. 不建议继续扩展的方向

为了保证和答辩主线一致，当前阶段不建议优先继续扩展这些现有分支：

- `CAS`
- `CMA`
- `PGI`
- `DBB`
- `RDBB`
- `DeepDBB`
- `EFM`

原因：

- 它们不属于答辩中明确承诺的三项改进
- 继续沿这些方向扩展会分散实验精力
- 会导致论文主线和代码主线不一致

---

## 7. 优先级建议

推荐优先级如下：

### P0

- 固化 baseline
- 建立标准实验配置

### P1

- 接入 A2C2f

### P2

- 实现 BiFPN

### P3

- 实现 SR 分支

### P4

- 组合 ALL
- 补齐最终文档和实验总结

---

## 8. 推荐的近期行动

建议接下来按下面顺序实施：

1. 新建 `yolo11-RGBT-baseline.yaml`
2. 新建 baseline 的 train / val / predict 配置
3. 新建 `yolo11-RGBT-a2c2f.yaml`
4. 跑通 baseline 和 `+A2C2f`
5. 开始实现 `BiFPN`

原因是这条路径最稳，也最接近答辩主线。

---

## 9. 最终交付标准

当以下条件都满足时，可以认为仓库已经基本实现答辩内容：

- 有明确的 `YOLOv11_RGBT baseline`
- 有 `+BiFPN`、`+A2C2f`、`+SR`、`+ALL` 四个模型版本
- 每个版本都有对应 YAML 和配置文件
- train / val / predict 都能正常运行
- 能输出可复现的实验指标
- README 和实验文档能清楚说明每项改进对应的代码位置和使用方式

