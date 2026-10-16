# 使用说明

本工具包提供 OpenFWI 声学正演与伴随算子、无条件 EDM 扩散先验训练、
四个数据集的预训练权重和无条件生成示例。每个数据集附一个速度模型及其观测。

克隆仓库、安装依赖并下载资源：

```bash
git clone https://github.com/minchen073/OpenFWI-Prior-Toolkit.git
cd OpenFWI-Prior-Toolkit
python -m pip install -e .
python -m pip install 'cupy-cuda12x>=13,<14'
python scripts/download_assets.py
python scripts/verify_assets.py
```

请先安装与 GPU 兼容的 PyTorch。声学算子要求 NVIDIA CUDA GPU；
无条件生成也支持 `--device cpu`。完整训练数据需自行准备。

四份预训练权重和四个示例包通过 GitHub Release 提供，总计约 394 MiB。
下载脚本会校验 SHA-256，并跳过已经下载且校验一致的文件。

无条件生成：

```bash
python scripts/sample.py \
  --checkpoint assets/checkpoints/FlatVel-B_EDM_ema_400k.pt \
  --num-samples 4 --steps 40 --seed 42 --output outputs/flatvel_samples
```

更换权重文件即可使用其他数据集。输出包括速度数组、图片和采样参数。
生成过程不使用地震观测，结果表示无条件先验样本。保存的数值未经截断；
图像色标范围为 1500–4500 m/s。

正演与伴随调用：

```bash
python scripts/acoustic_example.py \
  --example assets/examples/FlatVel-B.npz --output outputs/acoustic
```

训练前检查数据：

```bash
python scripts/train.py --config configs/FlatVel-B.yaml \
  --data-root /path/to/FlatVel-B --dry-run
```

去掉 `--dry-run` 即开始训练。短流程检查可加
`--steps 2 --batch-size 2 --num-workers 0`，该检查不能代替完整训练。
配置文件明确列出训练与监测所需文件。所有相对路径以当前工作目录为基准。
标准配置使用 400,000 步，发布权重为该固定终点的 EMA 参数。

完整的安装要求、数据文件格式、续训方式和算子约定见
[英文 README](../README.md)、[训练协议](TRAINING.md)和[算子说明](OPERATORS.md)。
