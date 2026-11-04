# R27便携推理与T4端到端诊断

只用于固定模型推理和计时，不训练、不联网下载权重、不提交平台。与R16便携工具复用相同stdlib输入/计时契约，模型及精度策略全部独立。本地仅进行文件准备及轻量检查，不导入Torch/加载权重。

## 三种显式模式

- captain-original：队长R27原图，512全图/384局部，定位器TF32开启。平台98.56为用户报告；已知定位热图近并列及整数边界存在跨GPU敏感性，不能承诺跨GPU逐图概率完全相同。
- stable-flip384：同一原权重，单次全图水平翻转，定位器TF32关闭；平台98.53为用户报告。
- detail-flip512：固定epoch3细节512局部权重，定位器TF32关闭，沿用已全量逐像素/预测一致的省复制并行流程。没有采用定位优先调度原型。valid AUROC0.990294；平台未反馈。完整4090D平均116.48ms/p95176.01ms，尚未通过100ms/T4验收，仅支持此前已验证的JPEG路径。

不能将这些模式混用。None模式不存在；--mode必须明确给出。测试CSV行序和文件名仅从外部官方CSV的Image_name读取，忽略标签和年龄。原图与稳定384模式可读JPG/JPEG/PNG；细节512拒绝PNG，不静默换成不同解码流程。

## 准备私有便携目录

从仓库根目录运行，输出须为新目录：

```bash
python experiments/r27_inference/prepare.py \
  --materials-root /path/to/original_r27_materials \
  --detail-checkpoint /path/to/fixed_detail512_epoch3/best.pt \
  --output /path/to/new/r27_bundle
```

省略--detail-checkpoint时仅包含两种384局部模式。准备阶段只核验哈希/复制文件，不反序列化模型。复制固定inference/源码/三份原权重及所需适配器，不复制metadata、图片、valid标签、预测明细或训练工程。带细节512时仍保留原局部权重以支持三种模式，模型原文件合计约254.82MB；仅原R27约143.48MB，均小于500MB。bundle.json记录源码/权重SHA与字节数；运行前后核验，禁止改名替换权重冒充其他候选。便携目录含私有权重，不公开上传。

## 环境与预测

目标环境为Linux，Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128；CPU线程8，cuDNN benchmark关闭，运行时同样核验NumPy/Pillow固定版本。environment.yml记录已运行服务器对应的NumPy2.5.4/Pillow12.3.0。当前尚未从该YAML新建环境并安装验证，也未在真实T4运行，不把打包/CPU检查称作GPU验收。推理模型不下载通用预训练权重。

```bash
# 在新服务器上创建环境；若已具备匹配环境可直接使用。
conda env create -f environment.yml
conda activate r27-inference
# 进入便携目录（以下run.py属于便携包根目录）
python -u run.py predict --mode detail-flip512 \
  --images /path/to/test --metadata /path/to/test.csv \
  --output /path/to/new/predictions
```

predict固定要求5855行；输出submission.csv、逐图predictions.jsonl、report.json及最终complete.json。每图probability为有限0~1连续值，17位有效数字，阈值0.5。源数据仍由用户提供。运行失败保留failed.txt和部分明细，不写完成标记。未提供--reference时只说明成功运行，不证明与历史分数复现。

## T4计时与搬迁后复现

```bash
# 优先先做256图严格预测复现；reference为相同模式的历史trace JSONL。
python -u run.py benchmark --mode detail-flip512 \
  --images /path/to/valid --metadata /path/to/valid.csv \
  --reference /path/to/detail512_valid_trace.jsonl \
  --limit 256 --warmup 20 --output /path/to/new/t4_smoke
# 通过复现后，用新输出目录测完整valid；原图或384版另起进程。
python -u run.py benchmark --mode detail-flip512 \
  --images /path/to/valid --metadata /path/to/valid.csv \
  --reference /path/to/detail512_valid_trace.jsonl \
  --limit 0 --warmup 20 --output /path/to/new/t4_full
```

--reference只用于对照，不参与模型输入或选择。概率容差固定1e-7，并要求历史记录含有的ROI及裁剪RGB哈希一致；跨GPUFP16差异可能触发失败，需保留证据诊断，不自动放宽门限。captain-original的原TF32热图跨GPU问题仍在，不能用其他模式参考绕过。

在非T4卡诊断时显式增加--diagnostic；否则benchmark拒绝非T4。默认推理预算30分钟，可用--max-seconds调整（最大2小时），预算包含首次推理和预热、不含模型加载，不能中断正在执行的一次调用。不同模式分别新进程运行，避免GPU占用、缓存和状态干扰。

外层计时从GPU同步后开始，覆盖开JPEG/PNG、EXIF/RGB、翻转、全图预处理/分类、实时定位、ROI/局部分类和融合，以及现有诊断像素哈希及单图线程清理，结束时再GPU同步。不含模型加载、身份哈希、清单扫描、逐图trace/CSV写入、最后线程池shutdown；首次预测不等于清空OS缓存的冷盘测试。报告给出load/first、均值/p50/p95/p99/max、超过100ms数量、GPU/驱动/包版本/线程数、进程峰值RSS与CUDA峰值allocated/reserved、是否全覆盖、来源SHA和模型大小。official_acceptance始终false；hardware_is_t4和all_measured_under_or_equal_100ms仅描述本次实测，不猜测官方聚合口径。

本地检查：python experiments/r27_inference/test_cpu.py。只检查打包身份、输入顺序/路径、统计契约、CLI隔离和失败保护。当前尚未完成此便携入口的服务器真实模型集成检查；旧实验入口的GPU成功记录不能直接代替新入口验收。
