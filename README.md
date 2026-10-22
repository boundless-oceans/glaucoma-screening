# 青光眼图像与年龄融合训练

当前默认使用 EfficientNet-B2＋年龄融合，并开启保守去黑边。
仅 train 参与训练，valid 用于验证和选择权重，test 只做预测，不新增人工标注。
正式训练在 AutoDL 执行；本地只编辑代码及运行轻量检查。
遵循最新比赛通知，不使用医学类预训练权重；本实验从我们用通用 ImageNet 初始化、官方 train 训练所得的最佳权重继续微调。

## 文档目录

- [目录结构](#project-layout)
- [模型配置与对照实验](#model-config)
- [去黑边：预览、开关与检查点兼容](#black-border)
- [环境准备](#environment)
- [推荐运行顺序](#run-workflow)
- [最佳权重短程微调](#finetune)
- [后台运行与日志](#logging)
- [检查点与预测结果](#checkpoints)
- [训练方案与验证范围](#validation)
- [代码测试与版本控制](#development)

<a id="project-layout"></a>

## 目录结构

```text
/root/autodl-tmp/
├── glaucoma_first/                       # 数据压缩包解压后的数据根目录
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── training/                             # 代码仓库根目录
    ├── configs/
    │   ├── train.json                    # B2 正式训练，默认开启去黑边
    │   ├── smoke.json                    # B2 小样本，继承默认去黑边配置
    │   ├── finetune.json                 # 最佳权重微调，1e-5、3轮
    │   ├── finetune_smoke.json           # 保持320输入的小样本微调
    │   ├── no_blackborder.json           # B2 关闭去黑边的对照配置
    │   ├── resnet18.json                 # ResNet18 全量对照，不去黑边
    │   └── resnet18_smoke.json           # ResNet18 小样本，不去黑边
    ├── src/
    │   └── glaucoma_training/
    │       ├── __init__.py               # 项目目录常量，不加载 PyTorch
    │       ├── train.py                  # 训练、验证与断点恢复
    │       ├── finetune.py               # 初始权重兼容性和基线复验保护
    │       ├── evaluate.py               # 最佳模型验证及测试集预测
    │       ├── predict.py                # 仅生成测试集预测的独立入口
    │       ├── common.py                 # 可选骨干、Dataset 和推理逻辑
    │       ├── model_config.py           # 配置合并、架构识别和实验目录
    │       ├── preprocess_config.py      # 版本化预处理参数及旧权重默认值
    │       ├── preprocessing.py          # 共享去黑边与缩放补边逻辑
    │       ├── preview_preprocessing.py  # CPU 生成裁剪前后对照图
    │       ├── download_weights.py       # 在服务器预下载官方预训练权重
    │       ├── data_utils.py             # CSV、年龄处理与指标
    │       ├── run_utils.py              # 配置检查、输出保护与进程锁
    │       ├── preflight.py              # 上传后数据路径检查
    │       ├── gpu_check.py              # GPU 前向/反向检查
    │       └── launch.py                 # Shell 调度、日志与状态记录
    ├── scripts/
    │   ├── train.sh                      # 训练和续训入口
    │   ├── test.sh                       # 模型评估和预测入口
    │   ├── run_checks.py                 # 轻量测试，不导入PyTorch
    │   └── check_finetune.py             # 服务器显式运行的微型模型集成检查
    ├── tests/                            # 自动化代码测试，不训练模型
    ├── docs/                             # 本地改进计划，Git 忽略，克隆后不自带
    ├── outputs/                          # 预处理预览等生成产物，不入 Git
    ├── logs/                             # 自动生成的日志，不入 Git
    ├── runs/                             # 自动生成的权重和预测结果，不入 Git
    ├── environment.yml                   # 服务器 Conda 环境
    ├── requirements.txt                  # 使用现有镜像时补装的依赖
    ├── README.md
    ├── .gitignore
    └── .gitattributes
```

项目根目录保留说明和环境配置；模型超参数放 configs，Python 业务源码放 src。
evaluate.py 是实际运行模型的入口，不是 tests 中的开发测试。
数据和代码可以分开上传。--data-root 指向直接包含三个图片目录和三个 CSV 的目录，不能指向压缩包。
代码按不区分大小写的文件名匹配原图，兼容 CSV 中 .jpg 与磁盘文件 .JPG 的差异。
请上传完整新目录，不再运行旧版根目录中的 train.py / launch.py，也不要只上传零散文件。

<a id="model-config"></a>

## 模型配置与对照实验

支持 model_name=resnet18 或 efficientnet_b2，默认使用后者。
EfficientNet-B2 仍与标准化年龄、年龄缺失标记拼接分类；预训练权重来自 torchvision 官方实现。
去黑边对照只改变预处理开关：正式输入仍为 320×320、batch=32，图像增强、BCE 权重和学习率保持一致。
比较架构时请使用 configs/no_blackborder.json 对照原 ResNet18，避免同时更换骨干和预处理后无法归因。
不预设新架构一定更好，需要用同一验证集、相近训练预算比较；T4 延迟也尚未测量。

默认目录按架构隔离：

| 配置 | 模型 | 默认实验目录 |
| --- | --- | --- |
| configs/train.json | EfficientNet-B2 | runs/efficientnet_b2_blackborder |
| configs/smoke.json | EfficientNet-B2 小样本 | runs/efficientnet_b2_blackborder_smoke |
| configs/no_blackborder.json | EfficientNet-B2 不去黑边 | runs/efficientnet_b2 |
| configs/resnet18.json | ResNet18 对照，不去黑边 | runs/baseline |
| configs/resnet18_smoke.json | ResNet18 小样本对照 | runs/smoke |

同一分支也可以运行 ResNet18 对照，不必反复切分支：

```bash
bash scripts/train.sh --config configs/resnet18_smoke.json
bash scripts/test.sh --smoke --run-dir runs/smoke
```

--smoke 和 --config 不能同时传入；小样本测试按检查点实际架构运行。

全量对照则使用：

```bash
bash scripts/train.sh --config configs/resnet18.json --max-hours 3
bash scripts/test.sh --config configs/resnet18.json
```

显式 --checkpoint 时，实际模型以权重内记录为准；--config 控制默认实验目录和配置日志，不会改变权重的架构。
不要使用 B2 的默认命令续训旧 runs/baseline，恢复旧 ResNet18 任务时需传 configs/resnet18.json。

<a id="black-border"></a>

## 去黑边：预览、开关与检查点兼容

当前默认流水线：原图 EXIF/RGB 校正 → 可选去黑边 → 等比例缩放补边 → 训练时随机增强 → 模型。
训练、验证、独立评估和测试预测共用 prepare_image；验证和测试不做随机增强。
配置中记录 remove_black_border、black_border_threshold、black_border_margin、black_border_version，随权重保存。
旧权重缺少这些参数时按不开启处理；评估按权重配置执行，不会套用当前分支的新默认值。
改变开关或参数后不允许续用旧实验目录的检查点，新默认目录包含 _blackborder，防止和原 B2 任务混用。

version=1 是保守的外接矩形裁剪：

- 在最长边不超过 512 的缩小图上检测 max(R,G,B)>8 的前景，取全部前景的外接框。
- 默认保留原图宽高各 2% 的安全边距，不丢弃小连通块来追求更紧的裁剪。
- 暗图、前景过少、裁剪区域过小等情况回退整张原图；这是安全启发式，不能保证所有图片都判断正确。
- 不裁视盘、不做分割，不保证清除圆形视野四角的所有黑色；后续为统一尺寸补上的黑边也会保留。
- 不覆盖原始图片、不新增标注，不引入 OpenCV 等新依赖。检测及解码耗时仍需计入 T4 端到端延迟。

先检查训练图的前后对照，不加载 PyTorch、不训练：

```bash
cd /root/autodl-tmp/training
conda activate glaucoma-cloud
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m glaucoma_training.preview_preprocessing --count 12 --split train
```

输出到 training/outputs/preprocess_preview_时间戳/preview.jpg 和 report.json。
每对左图为原图（绿框表示拟保留区域），右图为模型输入预览；JSON 记录原图坐标、裁剪原因和参数。
默认包含正负样本的小样本预览，也可 --split valid。预览不提供 test 选项，不以测试图挑选预处理参数。
outputs 已被 Git 忽略，里面的私有图片不应上传代码仓库。

正式训练和测试仍是原来的两个 Shell 命令，默认开启去黑边：

```bash
bash scripts/train.sh --smoke
bash scripts/test.sh --smoke
bash scripts/train.sh --max-hours 3
bash scripts/test.sh
```

与不去黑边的 B2 对照：

```bash
bash scripts/train.sh --config configs/no_blackborder.json --max-hours 3
bash scripts/test.sh --config configs/no_blackborder.json
```

开启后的默认目录为 runs/efficientnet_b2_blackborder（小样本加 _smoke）；关闭时仍为 runs/efficientnet_b2。
实际提交文件所在 evaluations 子目录由测试脚本打印。不要只根据预览认为效果提高，仍需比较验证指标。
本地已对 8 张真实训练图片生成预览：7 张裁掉外围背景，1 张保留整图；这是样例检查，不是全量质量保证。

<a id="environment"></a>

## 环境准备

无卡模式也可以创建环境；有卡开机后才能验证 CUDA。

```bash
cd /root/autodl-tmp/training
conda env create -f environment.yml
conda activate glaucoma-cloud
```

配置为 Python 3.12、torch 2.8.0+cu128、torchvision 0.23.0+cu128，固定 NumPy 与 Pillow 版本。
若激活提示 shell 未初始化，先运行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。
不要使用本地项目父目录中为 Windows/4060 准备的 cu126 环境。
若沿用镜像现有 GPU 环境，也可只安装 requirements.txt；Shell 默认使用 glaucoma-cloud，直接 Python 模块命令则使用当前环境。

Shell 会自动设置 PYTHONPATH，无需 pip 安装本项目。需要直接执行 Python 模块时，在代码根目录先执行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m glaucoma_training.preflight --data-root /root/autodl-tmp/glaucoma_first
```

应看到 train=51396、valid=5849、test=5855，所有图片路径匹配。
preflight 不加载 PyTorch、不调用 GPU，也不完整解码图片，不能代替数据完整性/传输哈希检查。
首次需要下载官方 ImageNet 权重，可提前在当前环境下载，避免占用 GPU 时段：

```bash
python -m glaucoma_training.download_weights
```

此命令不训练、不调用 GPU。下载失败先解决网络，不要将 pretrained=false 当作正式替代方案。

<a id="run-workflow"></a>

## 推荐运行顺序

有卡开机后，在代码根目录运行：

```bash
conda activate glaucoma-cloud
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m glaucoma_training.gpu_check
bash scripts/train.sh --smoke
bash scripts/test.sh --smoke
```

小样本使用 256 张训练图、128 张验证图、16 张测试图、1 轮，输出 DEBUG_predictions.csv，不能提交比赛。
运行正常再开始正式任务：

```bash
bash scripts/train.sh --max-hours 3
bash scripts/test.sh
```

- 默认数据路径 /root/autodl-tmp/glaucoma_first，默认模型目录 training/runs/efficientnet_b2_blackborder。
- Shell 从自身位置找到代码和 configs，终端当前目录不影响默认路径；也可以使用脚本绝对路径启动。
- 手动传入的相对 --config、--run-dir、--checkpoint 仍相对于终端当前目录，推荐传绝对路径。
- train.sh 调用 launch，再启动训练模块；同一命令自动恢复 last.pt，不重复开始。
- train.sh 固定跳过训练后的测试集推理；test.sh 单独加载 best.pt，评估 valid 并生成 test 的预测。
- 至少完成一轮验证才会有 best.pt，未生成时测试入口会明确报错。
- --smoke 对应 configs/smoke.json 和 runs/efficientnet_b2_blackborder_smoke；正式配置为 configs/train.json，包含 model_name=efficientnet_b2。
- 可用 `bash scripts/train.sh --config /配置绝对路径.json --run-dir /新实验绝对路径` 建立独立实验，不能同时传 --smoke。
- 测试其他权重：`bash scripts/test.sh --checkpoint /权重绝对路径/best.pt`。

脚本自动激活已创建的 glaucoma-cloud；不会安装环境、申请 GPU、自动开关机或等待空闲 GPU。
--max-hours 默认 3，设为 0 表示不设预算。计时不含初始下载、初始化和测试；验证会完整结束，因此不是严格的关机定时器。
需要让出 GPU 时，前台按一次 Ctrl+C，或向后台 PID 发送 SIGTERM，等待保存提示后再关机。

<a id="logging"></a>

## 后台运行与日志

```bash
cd /root/autodl-tmp/training
mkdir -p logs
nohup bash scripts/train.sh --max-hours 3 >> logs/bootstrap_train.log 2>&1 &
echo $!
```

记住打印出的 PID；后台测试可用 `nohup bash scripts/test.sh >> logs/bootstrap_test.log 2>&1 &`。
bootstrap 日志记录 Conda 激活等早期失败；Python 成功启动后，会另建独立日志：

```text
logs/
├── latest_train.txt                       # 最近一次训练日志目录绝对路径
├── latest_test.txt                        # 最近一次评估日志目录绝对路径
└── train_UTC时间戳_PID/ 或 test_UTC时间戳_PID/
    ├── console.log                       # 标准输出、错误输出和异常堆栈，带时间戳
    ├── status.json                       # PID、阶段、起止时间、退出码和状态
    ├── environment.json                  # Python、依赖版本、GPU 可见设置
    ├── effective_config.json             # 合并默认值后的配置，测试时是目录配置
    └── requested_config.json             # 训练传入的配置原文，仅训练日志生成
```

启动时会把 Git 提交号、nvidia-smi、磁盘空间和内存情况写入 console.log。
每次启动独立目录，旧日志不会覆盖；子进程失败会返回非零退出码，Python 启用无缓冲和 faulthandler。
查看最新日志和状态：

```bash
tail -f "$(cat logs/latest_train.txt)/console.log"
cat "$(cat logs/latest_train.txt)/status.json"
cat runs/efficientnet_b2_blackborder/progress.json
```

每 25 个训练批次及每轮最后一批会打印进度、loss 和学习率，每轮结束打印验证指标。
每次检查点保存会打印保存位置；history.jsonl 另存每轮指标，不包含全部控制台文本。
paused 是正常保存暂停，completed 表示完成，failed 表示出错。状态卡在 running 不代表进程一定存活：
SIGKILL、系统 OOM 或主机掉电时来不及写最终状态，需要结合进程、最后日志、progress 和平台系统日志排查。
日志和权重写入时尽量刷新，但不能保证物理磁盘故障后可恢复，重要权重请另行备份。

<a id="checkpoints"></a>

## 检查点与预测结果

```text
runs/efficientnet_b2_blackborder/
├── last.pt                          # 模型、优化器、AMP、随机状态及下一批位置
├── best.pt                          # 最佳验证模型参数
├── progress.json                    # 最近一次成功保存的状态与位置
├── config.json                      # 本次生效配置快照，属于运行产物
├── environment.json                 # 训练运行环境
├── age_stats.json                   # 训练数据拟合的年龄统计量
├── history.jsonl                    # 各轮验证指标
├── best_metrics.json                # 最佳模型验证指标
├── valid_predictions.csv            # 最佳模型验证预测
├── valid_thresholds.csv             # 最佳模型的验证阈值表现
└── evaluations/
    └── UTC时间戳_PID/
        ├── validation_metrics.json  # 重新验证的指标
        ├── validation_predictions.csv # 验证预测与真实标签
        ├── checkpoint_info.json     # 权重 SHA256、环境、配置及完成状态
        └── submission.csv           # 本次正式预测结果
```

正式提交使用 test.sh 打印出的 evaluations 目录中的 submission.csv。
完整数据应有 5855 行记录，列为 Image_name,Probability,Pred_Label，UTF-8，无索引，保持 test.csv 原始顺序和名称。
测试集没有真实标签，不能计算测试集 AUROC/准确率；评估入口只报告 valid 的指标。
小样本权重默认不能生成正式提交，--smoke 输出 DEBUG_predictions.csv。
默认标签阈值 0.5，概率始终是 sigmoid 输出。FAQ 与官网对评分输入有冲突，核实前不自动优化提交阈值；
确认可调后可传 `bash scripts/test.sh --threshold 0.3`，应只依据验证集作决定。
选模主要看概率 AUROC，完全相等时比较梯形 AUPRC；同时记录 Average Precision，两者定义不同。
valid_thresholds.csv 记录 0.01～0.99 的验证阈值表现和 hard_label_auroc。不要把阈值搜索后的值当作独立泛化证明。

第一次训练前即保存初始 last.pt，此后每 200 批、训练阶段结束前、验证结束后和正常停止时保存。
正常暂停从下一批继续，强制停止只能恢复到最近一次已保存的批次。续训保持配置、CSV 和原图不变，
保留 last.pt 和 best.pt；最佳模型丢失会报错。不要删除 .run.lock 来并发启动同一实验，进程退出会释放系统锁。
旧版缺少 model_name 的检查点按 ResNet18 识别；新检查点保存架构标记。
评估按检查点选择骨干，续训拒绝将 ResNet18 权重载入 EfficientNet-B2；相同架构的旧权重键名保持兼容。

<a id="validation"></a>

## 训练方案与验证范围

configs/train.json 默认：320×320，batch=32，4 个读取进程，12 轮，AdamW，学习率 1e-4，混合精度。
图像先按配置保守去黑边，再等比例缩放补边，训练时 50% 水平翻转、±10° 旋转、轻微亮度/对比度增强；验证与预测不随机增强。
年龄空值、非有限值、<=0 或 >110 视为缺失，用训练数据中位数填补；均值和标准差仅从训练数据计算。
图像特征与年龄、缺失标记拼接分类，默认 BCE pos_weight=1，不同时叠加多种不平衡处理。
显存不足时在新实验配置中减小 batch_size；内存/数据加载问题可减小 num_workers。
非零样本 limit 仅用于流程调试。微调配置 pretrained=false 表示不下载 ImageNet 初始化，必须从 --init-checkpoint 加载完整任务权重。配置改变请建新实验目录。
原图解码可能成为性能瓶颈，跑通后再按 GPU 利用率决定是否建立缩放缓存。

原版于2026-10-19在5090完成12轮，最佳第5轮AUROC为0.981308；次日独立评估复现。
本次微调功能仅完成轻量检查，未在本地加载PyTorch或使用GPU；真实模型微调及恢复仍待服务器集成检查。
数据顺序与增强按 seed、epoch、图片索引固定，跨 GPU/驱动/PyTorch 版本不保证逐位相同。
最终仍需官方 PyTorch 2.10.0/torchvision 0.25.0、T4 16GB 环境核验，单图端到端 <=100ms 和模型 <=500MB 尚未实测。
仅加载自己生成的可信权重。复赛按届时清单打包，不能把 last.pt 等整个 runs 目录直接当作提交模型包。

后续早停、错误样本报告及不平衡处理实验仍在计划中，当前版本尚未实现早停和错误图片报告。
本地计划位于 docs/，该目录被 Git 忽略，不随仓库发布。

<a id="finetune"></a>

## 最佳权重短程微调

这是从 main 基线出发的独立实验，不启用分组学习率、正类加权、EMA或TTA。模型结构、320输入、去黑边、年龄处理、AMP及增强保持原版；学习率1e-5，统一AdamW，3轮余弦衰减。优化器和scaler重新初始化，训练轮数从0计；不是恢复原版last.pt。
微调默认目录为 runs/efficientnet_b2_blackborder_finetune，小样本为 runs/efficientnet_b2_blackborder_smoke_finetune，不覆盖来源模型。

先在服务器激活glaucoma-cloud环境，运行合成数据集成检查（它会真正训练微型ResNet18，禁止下载预训练权重）：

```bash
python scripts/check_finetune.py --device cuda
```

检查包含：源模型训练、微调前基线复验、预算暂停、恢复、保留较优模型，以及加载最佳权重生成预测。独立产物位于 outputs/check_finetune_<时间戳>/。可显式使用 --device cpu 在服务器无卡模式检查，不要在本地电脑运行模型集成脚本。

以下假设旧版 best.pt 已保存在服务器这个路径；文件缺失时先从备份恢复，不要拿 last.pt 或分组/加权实验代替：

```bash
bash scripts/train.sh --config configs/finetune_smoke.json \
  --init-checkpoint /root/autodl-tmp/training/runs/efficientnet_b2_blackborder/best.pt
bash scripts/test.sh --smoke --run-dir runs/efficientnet_b2_blackborder_smoke_finetune
```

小样本只验证流程，保留320输入和batch32；子集AUROC不与完整0.9813比较，initial_validation.json明确记为subset_only。
正式微调：

```bash
mkdir -p logs
nohup bash scripts/train.sh --config configs/finetune.json \
  --init-checkpoint /root/autodl-tmp/training/runs/efficientnet_b2_blackborder/best.pt \
  --max-hours 3 >> logs/bootstrap_finetune.log 2>&1 &
```

第一步完整评估来源模型，AUROC及AUPRC默认必须与来源权重的best_metrics相差不超过1e-5，否则在优化器更新前停止。直接train模块的 --baseline-tolerance 仅允许0至0.001，排查差异前不要放宽门槛。
源权重必须为完整数据模型，架构、数据CSV签名、输入和预处理必须兼容；年龄统计沿用源权重，完整分类头严格加载，不会重置为随机分类头。
来源路径、SHA256、来源轮数及指标记录在last.pt、best.pt和initial_validation.json。第0轮先保存复验后的源模型为保底best；只有(AUROC,AUPRC)更优才替换。新best.pt的epoch表示微调轮数，0表示仍是初始模型，原轮数见initialization.source_epoch。
best_manifest.json最后写入并记录best四个产物的哈希；微调恢复会检查完整性，损坏时明确报错，不悄悄继续。

预算默认仍为3小时。完成返回0，到时断点保存暂停返回3，用户中断返回130。暂停后原命令可续训；也可以省略 --init-checkpoint，仅用新实验last.pt恢复。再次提供初始权重时必须与记录哈希一致，不能在续训时更换来源。

正式评估：

```bash
bash scripts/test.sh --config configs/finetune.json
# 无卡：显式使用CPU，不更改模型权重；float32与CUDA AMP概率可能略有差异
bash scripts/test.sh --config configs/finetune.json --device cpu --workers 0
# 按需只评估验证集；默认仍为valid+test
bash scripts/test.sh --config configs/finetune.json --splits valid
```

完整评估目录最后写入manifest.json，记录来源、权重及预测哈希。只跑test不会重新计算valid指标，checkpoint_validation是保存时指标，不是本次复验结果。以test.sh打印的evaluation目录为准，避免拿错旧submission。
CUDA/CPU真实推理兼容、短程微调是否提升以及T4端到端延迟，都需要服务器实测，轻量替身测试不能替代它们。

<a id="development"></a>

## 代码测试与版本控制

本地在 training 根目录执行以下轻量测试，不会训练模型：

```bash
python -B scripts/run_checks.py
```

Git 仅位于 training/.git；数据、docs、outputs、logs、runs、缓存均不入库。提交信息用 `关键词: 改动总结`，分支用 `关键词/用途`。
模型与私有数据不要公开上传，服务器不必携带 .git；需要 Git 记录诊断时再同步私有代码仓库。
