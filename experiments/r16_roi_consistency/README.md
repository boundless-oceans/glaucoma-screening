# R16 ROI一致性独立实验入口

承接队长R16，而非把我方B0替换成ConvNeXt后称为R16。复用原E32 epoch7全图权重，只准备并训练局部。此目录可放在training/experiments/r16_roi_consistency/进行Git管理；运行项目必须准备到另一个新目录。

## 变体范围

固定原B2/512全图、ConvNeXtTiny/384局部、ImageNet22K→1K初始化、无年龄、pos_weight≈30.0175、backbone/head学习率1e-4/1e-3、一轮warmup后按成功更新步cosine、原增强、融合AUROC再AP选模、50/50 raw-logit融合和global fallback。

变化限定在ROI一致性：归一化坐标CSV用17位有效数字写出并用round_trip读回；train与valid都按B1生成冻结CAM资产，以对应原图实时B1路径。不是只改CSV一行就保证所有推理数值不变；train资产的batch32→B1也是本次明确记录的变化。原R16材料保持原样。

vendor/保留最小必需历史源码，不再分发官方逐行数据清单；SOURCE_SHA256.json绑定当前分发的LF文本，UPSTREAM_SHA256.json记录历史原包字节哈希（包括不分发的数据文件），DATA_PROVENANCE.json仅保留数据文件哈希和大小。scripts/prepare.py对新运行项目应用显式窄补丁，并记录新源码指纹。没有复制医学模型研究模块，没有引入额外数据。

## 当前验证状态

2026-10-30已完成服务器真实B2 CAM、32图裁剪像素及随机局部更新检查，随后完成正式局部训练与完整原图B1验证。训练15轮、最佳第10轮，最终AUROC为0.9890619593、AP为0.8517480648，未超过队长原版的0.9893770392与0.8645259663。本轮不替换原版基线，也未生成test提交。代码仍按显式阶段执行，不会自动开始全图训练或提交。

服务器已在独立的/root/r16-roi-env环境安装并验收torch==2.10.0+cu128、torchvision==0.25.0+cu128；代码会拒绝其他版本。requirements-training.txt保留原作者环境记录。原glaucoma-cloud的2.8/0.23环境未覆盖；T4端到端验收仍未完成。

## 本地数据生成与公开说明

prepare.py调用scripts/manifest_data.py读取本地官方train.csv/valid.csv，按图像ID排序，生成与原版逐字节一致的角色清单，写到新运行项目manifests/。会检查51396/1657、5849/191计数、文件名/标签/年龄格式及历史SHA256；输入错版、重复ID或数据更改均拒绝继续。原始解压图像保持原大写.JPG文件名；生成清单不扫描或解码图像。

7项CPU配方检查使用tests/synthetic_data.py人工生成的身份、年龄和标签，不读取真实样本清单。已用本地真实官方CSV验证train与valid生成结果和历史哈希一致；本地两种准备入口及旧审计绑定均通过，不反序列化模型。

旧实验项目和归档保持原样。源码当前删除不会从Git历史抹去清单，公开前仍需使用干净快照集成；不要将含原清单历史的开发分支推到公开仓库。

## 1. CPU检查（需要NumPy、Pillow、pandas）

```bash
python tests/test_recipe.py
```

## 2. 准备独立运行项目（纯文件操作，不训练）

```bash
python scripts/prepare.py --output /root/autodl-tmp/r16_roi_consistency \
  --data-root /root/autodl-tmp/glaucoma_first \
  --global-checkpoint /path/to/E32-r16-official-global-20261026-222714/best.pt
```

全图权重必须保留best.pt文件名，哈希必须为b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2。不能拿我方B2或其他轮次代替。准备时可选--local-initializer，但几何检查阶段不需要初始化权重文件。

## 3. 服务器小样本工程验收

在目标独立环境中，从新运行项目执行：

```bash
cd /root/autodl-tmp/r16_roi_consistency
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p logs
python -u run.py check > logs/check.log 2>&1
```

check使用固定的16阳性＋16阴性train图，实时计算冻结E32 B1 CAM，经历CSV写入/读回，再比较共享解码与单独解码的裁剪像素。最后随机初始化一个ConvNeXtTiny，做两个4图batch的更新，检查实际前向/反向；随机模型随后丢弃，不保存为正式模型，不下载预训练。不访问test；检查结果绑定源码、CSV、权重与运行环境。

## 4. 检查通过后再生成新ROI资产

```bash
nohup python -u run.py assets > logs/assets.log 2>&1 < /dev/null &
```

资产覆盖train/valid，不包含test；默认最大3600秒，因B1代替原train batch32而扩大原1200秒安全预算，实际耗时待测。超时失败，不自动重试/继续。此阶段不训练局部；看到ASSETS_COMPLETE并验证完成标记后再进入下一步。

## 5. 正式局部训练是单独、显式的后续操作

原通用ConvNeXt初始化文件SHA256必须为f22f8850dcded245f4ae3bc402bbde9c6f5e4e23e76dfaea3aadb75aa9d3705a。它是ImageNet22K→1K/384，不是torchvision默认ImageNet1K权重。

```bash
nohup python -u run.py train \
  --local-initializer /path/to/convnext_tiny_22k_1k_384.pth \
  > logs/train.log 2>&1 < /dev/null &
```

保持原R16最多20轮、patience5、局部90分钟预算。此独立入口保留R16 one-shot语义，**不支持我方旧入口的中间batch精确恢复**。训练开始后不会隐式重跑；中断或预算耗尽会保留已有best/last/partial及记录，需要人工评估。不要把原来的bash scripts/train.sh --max-hours 3命令套进来。

## 6. 完整验证是另一条显式操作

```bash
python -u run.py evaluate > logs/evaluate.log 2>&1
```

使用固定E32及本次局部best走原图实时CAM B1路径评估官方valid，不生成test预测或自动提交。只有正式对照通过后，才决定后续测试导出。仍需单独的T4端到端验收，不能用4090测试宣称达标。
