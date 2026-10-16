# Physical-State-Guided Diffusion Sampling for Full-Waveform Inversion

本仓库提供我们文章目前开放的实现模块：

- CuPy 声学正演与离散伴随梯度。
- 原生 PyTorch 正演与自动求导梯度，无需 CuPy。
- 四个 OpenFWI 数据集的 EDM 先验训练脚本与配置。
- 每个数据集一个预训练 EMA 权重，以及一个速度模型与观测示例。
- EDM 18 步 Heun 和 VP 调度下的 1000 步 DDIM 无条件生成。

完整 PSG 反演代码与实验复现流程将在文章发表后提供。

## 安装

```bash
git clone https://github.com/minchen073/OpenFWI-Prior-Toolkit.git
cd OpenFWI-Prior-Toolkit
python -m pip install -e .
python scripts/download_assets.py
python scripts/verify_assets.py
```

请先安装与 GPU 兼容的 PyTorch。若使用 CuPy 正演／伴随后端，还需安装：

```bash
python -m pip install 'cupy-cuda12x>=13,<14'
```

四份权重和四个示例包通过 Release 提供，总计约 394 MiB。下载脚本校验
SHA-256，并跳过已下载且校验一致的文件。完整训练数据需自行准备。

## 无条件生成

EDM 18 步 Heun（35 次去噪器调用）：

```bash
python scripts/sample.py \
  --checkpoint assets/checkpoints/FlatVel-B_EDM_ema_400k.pt \
  --sampler edm-heun --steps 18 --num-samples 4 --seed 42 \
  --output outputs/flatvel_edm_heun
```

VP 1000 步 DDIM（1000 次去噪器调用）：

```bash
python scripts/sample.py \
  --checkpoint assets/checkpoints/FlatVel-B_EDM_ema_400k.pt \
  --sampler vp-ddim --steps 1000 --num-samples 4 --seed 42 \
  --output outputs/flatvel_vp_ddim
```

省略 `--steps` 时，两个采样器分别默认使用 18 和 1000 步。更换权重文件
即可使用其他数据集。两个采样器都使用相同的 EDM 先验；VP DDIM 先将
VP 状态转换到 EDM 坐标，再执行确定性 DDIM 更新（eta=0）。VP 调度为
`beta(t)=0.1+19.9t`，1000 个求值时刻从 1 到 0.001，最后到达 0。
VP 起点对应的 EDM 噪声约为 152.17，不截断到 Heun 的 80。

输出包括速度数组、图片和参数。生成过程不使用地震观测；保存的数值未经
截断，图像色标为 1500–4500 m/s。公式见[采样说明](SAMPLING.md)。

## 正演与梯度

```bash
python scripts/acoustic_example.py --backend cupy --device cuda:0 \
  --example assets/examples/FlatVel-B.npz --output outputs/acoustic_cupy

python scripts/acoustic_example.py --backend torch --device cuda:0 \
  --example assets/examples/FlatVel-B.npz --output outputs/acoustic_torch
```

`torch` 后端完全通过原生 PyTorch 算子和自动求导计算梯度，也可用
`--device cpu`。默认每 20 个时间步使用激活检查点，可用 `--block-size`
调整内存与重计算开销。两个后端使用相同的离散格式，输出与梯度可直接对照。
完整网格建议使用 GPU。

## 训练

```bash
python scripts/train.py --config configs/FlatVel-B.yaml \
  --data-root /path/to/FlatVel-B --dry-run
```

去掉 `--dry-run` 即开始训练。短流程检查可加
`--steps 2 --batch-size 2 --num-workers 0`，该检查不能代替完整训练。
配置文件列出训练与监测文件。相对路径以当前工作目录为基准。
标准配置使用 400,000 步，发布权重为固定终点的 EMA 参数。

完整说明见[英文 README](../README.md)、[训练协议](TRAINING.md)和
[算子说明](OPERATORS.md)。
