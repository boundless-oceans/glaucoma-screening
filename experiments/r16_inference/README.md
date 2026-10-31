# 原版R16便携推理与T4计时

这套入口只做原图或单次水平翻转的冻结推理，不训练、不拟合年龄或阈值，不自动下载、提交平台或判定官方验收。保留原版E32 epoch7全图＋E33 epoch15局部权重、512/384输入、FP16、实时CAM/ROI及视图内logit融合。单次翻转直接复用已验证的r16_tta/imaging.py和pipeline.py，原图调用未修改的R08Pipeline。

## 准备便携目录（本地也允许，仅文件操作）

从本实验目录执行，原版权重路径可任意放置，输出必须为新目录：

```bash
python scripts/prepare.py --output /path/to/new/r16_bundle \
  --global-checkpoint /path/to/original_global.pt \
  --local-checkpoint /path/to/original_local.pt
```

需要相邻r16_roi_consistency/vendor与r16_hflip_tta/src目录。脚本检查固定源码及两份原版权重SHA256，将所需源码、许可证与权重复制到新目录，写出bundle.json。不会反序列化模型。运行目录不依赖原训练项目路径、角色清单、验证标签或历史审计，只有自身src/和model/；不含官方数据或Git历史。准备后的权重属于私有本地材料，不要将整个bundle上传公开仓库。

目标环境：Linux、Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128、CUDA GPU。NumPy、Pillow和pandas是当前原推理源码的依赖；4090D已验证环境为NumPy2.5.4/Pillow12.3.0/pandas3.0.6。模型入口强制检查Python/PyTorch/torchvision版本，记录其他包和驱动实际版本，不自动安装依赖。不需要scikit-learn或训练依赖PyYAML。

## 仅预测

进入新bundle目录执行：

```bash
python -u run.py predict --view single-flip \
  --images /path/to/official/test --metadata /path/to/official/test.csv \
  --output-dir /path/to/new/predictions
```

--view也可选original；无默认值，避免误用方案。默认要求5855行，可显式--expected-count指定其他已知数量。元数据只读取Image_name，保持官方顺序，兼容文件名大小写和JPG/JPEG/PNG；PNG使用原有Pillow路径解码，JPEG draft对PNG不生效，不将PNG耗时视为与JPEG相同。图像和元数据均由使用者提供。输出submission.csv采用17位有效数字及固定0.5标签，完成标记记录哈希。输出目录必须不存在。此命令可以读取带标签的CSV作为图片列表，但不会使用标签或计算指标；用户需选择正确的提交数据，不把验证集预测当成测试提交。

## 计时诊断

```bash
# 先用固定种子的256图试跑；不会生成submission.csv
python -u run.py benchmark --view single-flip \
  --images /path/to/official/valid --metadata /path/to/official/valid.csv \
  --limit 256 --warmup 20 --output-dir /path/to/new/smoke_benchmark

# 试跑核对输出后，目标T4上做完整覆盖；原图对照用另一个新进程/输出目录
python -u run.py benchmark --view single-flip \
  --images /path/to/official/valid --metadata /path/to/official/valid.csv \
  --limit 0 --warmup 20 --output-dir /path/to/new/full_benchmark
```

batch固定为1，torch CPU线程固定8，cudnn.benchmark保持关闭。先记录模型构建/加载和GPU迁移时间，再单独计第一次预测，然后预热。稳态每图在GPU同步后开始，经过原JPEG开文件、解码、EXIF/RGB、可选翻转、去黑边/补边、全图模型、CAM、ROI/局部模型和最终概率，最后GPU同步再停表。不把模型加载、元数据扫描、源文件/权重哈希及CSV写入算入稳态时延；首次预测并非清空OS缓存的冷盘测试。两种view用不同进程运行，固定顺序与OS缓存可能影响比较；不要与训练或其他测速同时运行。

报告包括模型加载ms、首次预测ms、稳态均值/p50/p95/p99/最大值、超过100ms数量、是否覆盖全量、真实GPU与驱动/运行环境、峰值显存、来源和权重哈希。hardware_is_t4只识别硬件名称；all_measured_under_or_equal_100ms仅描述本次测量。official_acceptance始终为false，评委的预热、聚合方式与完整环境仍需对照，不能仅凭均值或非T4结果宣称官方达标。

默认每次推理预算30分钟（含首次预测/预热，不含加载/文件校验），超时保留部分输出并标失败，不自动重试。首次单次调用不可抢占。results.json和complete.json仅在完整运行后写入；predictions.csv含逐图概率与时延，可与此前归档结果对齐核对搬迁后的输出。

## 本地允许的检查

```bash
python tests/test_inference.py
```

只检验哈希、打包、元数据顺序、计时汇总与CLI；不导入PyTorch或使用本地GPU。无卡时只准备目录，暂不运行predict/benchmark。

## 2026-10-31服务器验证

代码快照dd970304c8203871f4c524e77174a54f32e45e81已在4090D上实际检查：先用35图运行predict两种视图，再分别在新进程执行完整5849图valid benchmark（batch1、8个CPU线程、20次预热）。两种视图都与2026-10-30同机配对归档逐图概率完全相同，硬标签变化为0；包括此前诊断过的CAM敏感样本。这里比较的是同机对照，不改变旧历史归档严格复现失败的结论。本次没有训练或导出新的test提交。

|视图|valid AUROC|AP|稳态均值ms|p95 ms|最大值ms|超过100ms张数|
|---|---:|---:|---:|---:|---:|---:|
|original|0.9893756512|0.8644753091|58.73|86.21|160.56|91/5849|
|single-flip|0.9901034351|0.8600360104|62.16|90.85|127.82|64/5849|

模型加载分别约1.31/1.04秒，首次预测约9.38/8.28秒，均与稳态计时分开记录。权重合计142590240字节；执行前后核对固定权重和源码哈希。平均耗时受运行状态和缓存影响，不将本次视图间差异解释为稳定速度排名。T4尚未测试，4090D上也未做到所有单图不超过100ms，不能宣称满足复赛时延门槛。

归档位于工作区outputs/training_results/20261031_r16_portable_inference/（不入Git），包含代码快照、逐图输出、日志、计时和完成标记，排除已有私有备份中的权重文件。下载后已校验压缩包和53个文件哈希，本地仅用CPU复算两种视图指标、顺序、逐图一致性与时延统计。verification.json和RESULTS.md记录复核结果；下一项实测是目标T4环境的完整计时。
