# 青光眼图像与年龄融合训练

默认模型为 ImageNet 预训练 ResNet18，融合标准化年龄和年龄缺失标记。
仅 train 参与训练，valid 用于验证和选择权重，test 只做预测，不新增人工标注。
正式训练在 AutoDL 执行；本地只编辑代码及运行轻量检查。

## 目录结构

```text
/root/autodl-tmp/
  glaucoma_first/                 数据压缩包解压后的数据根目录
    train/
    valid/
    test/
    train.csv
    valid.csv
    test.csv
  training/                       代码仓库根目录
    configs/
      train.json                  正式训练参数
      smoke.json                  小样本参数，只覆盖与正式配置不同的项
    src/glaucoma_training/
      __init__.py                 项目目录常量，不加载 PyTorch
      train.py                    训练、验证与断点恢复
      evaluate.py                 最佳模型验证及测试集预测
      predict.py                  仅生成测试集预测的独立入口
      common.py                   模型、Dataset 和推理逻辑
      data_utils.py               CSV、年龄处理与指标
      run_utils.py                配置检查、输出保护与进程锁
      preflight.py                上传后数据路径检查
      gpu_check.py                GPU 前向/反向检查
      launch.py                   Shell 调度、日志与状态记录
    scripts/
      train.sh                    训练和续训入口
      test.sh                     模型评估和预测入口
    tests/                        自动化代码测试，不训练模型
    enviroment.yml                服务器 Conda 环境
    requirements.txt              使用现有镜像时补装的依赖
    README.md
    .gitignore
    .gitattributes
    logs/                         自动生成，不入 Git
    runs/                         自动生成，不入 Git
```

项目根目录保留说明和环境配置；模型超参数放 configs，Python 业务源码放 src。
evaluate.py 是实际运行模型的入口，不是 tests 中的开发测试。
数据和代码可以分开上传。--data-root 指向直接包含三个图片目录和三个 CSV 的目录，不能指向压缩包。
代码按不区分大小写的文件名匹配原图，兼容 CSV 中 .jpg 与磁盘文件 .JPG 的差异。
请上传完整新目录，不再运行旧版根目录中的 train.py / launch.py，也不要只上传零散文件。

## 环境准备

无卡模式也可以创建环境；有卡开机后才能验证 CUDA。

```bash
cd /root/autodl-tmp/training
conda env create -f enviroment.yml
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
python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.DEFAULT.get_state_dict(progress=True)"
```

此命令不训练、不调用 GPU。下载失败先解决网络，不要将 pretrained=false 当作正式替代方案。

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

- 默认数据路径 /root/autodl-tmp/glaucoma_first，默认模型目录 training/runs/baseline。
- Shell 从自身位置找到代码和 configs，终端当前目录不影响默认路径；也可以使用脚本绝对路径启动。
- 手动传入的相对 --config、--run-dir、--checkpoint 仍相对于终端当前目录，推荐传绝对路径。
- train.sh 调用 launch，再启动训练模块；同一命令自动恢复 last.pt，不重复开始。
- train.sh 固定跳过训练后的测试集推理；test.sh 单独加载 best.pt，评估 valid 并生成 test 的预测。
- 至少完成一轮验证才会有 best.pt，未生成时测试入口会明确报错。
- --smoke 对应 configs/smoke.json 和 runs/smoke；正式配置为 configs/train.json。
- 可用 `bash scripts/train.sh --config /配置绝对路径.json --run-dir /新实验绝对路径` 建立独立实验，不能同时传 --smoke。
- 测试其他权重：`bash scripts/test.sh --checkpoint /权重绝对路径/best.pt`。

脚本自动激活已创建的 glaucoma-cloud；不会安装环境、申请 GPU、自动开关机或等待空闲 GPU。
--max-hours 默认 3，设为 0 表示不设预算。计时不含初始下载、初始化和测试；验证会完整结束，因此不是严格的关机定时器。
需要让出 GPU 时，前台按一次 Ctrl+C，或向后台 PID 发送 SIGTERM，等待保存提示后再关机。

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
  latest_train.txt                 最近一次训练日志目录绝对路径
  latest_test.txt                  最近一次评估日志目录绝对路径
  train_UTC时间戳_PID/ 或 test_UTC时间戳_PID/
    console.log                    带时间戳的完整标准输出、错误输出、异常堆栈
    status.json                    PID、运行阶段、开始/结束时间、退出码和状态
    environment.json               Python、依赖版本、GPU 可见设置
    requested_config.json          训练传入的配置原文
```

启动时会把 Git 提交号、nvidia-smi、磁盘空间和内存情况写入 console.log。
每次启动独立目录，旧日志不会覆盖；子进程失败会返回非零退出码，Python 启用无缓冲和 faulthandler。
查看最新日志和状态：

```bash
tail -f "$(cat logs/latest_train.txt)/console.log"
cat "$(cat logs/latest_train.txt)/status.json"
cat runs/baseline/progress.json
```

每 25 个训练批次及每轮最后一批会打印进度、loss 和学习率，每轮结束打印验证指标。
每次检查点保存会打印保存位置；history.jsonl 另存每轮指标，不包含全部控制台文本。
paused 是正常保存暂停，completed 表示完成，failed 表示出错。状态卡在 running 不代表进程一定存活：
SIGKILL、系统 OOM 或主机掉电时来不及写最终状态，需要结合进程、最后日志、progress 和平台系统日志排查。
日志和权重写入时尽量刷新，但不能保证物理磁盘故障后可恢复，重要权重请另行备份。

## 检查点与预测结果

```text
runs/baseline/
  last.pt                         续训状态：模型、优化器、AMP、随机状态、下一批位置
  best.pt                         最佳验证模型参数
  progress.json                   最近一次成功保存的状态与位置
  config.json                     本次生效配置快照（运行产物，不是源码配置）
  environment.json / age_stats.json
  history.jsonl / best_metrics.json
  valid_predictions.csv / valid_thresholds.csv
  evaluations/UTC时间戳_PID/
    validation_metrics.json       重新验证的指标
    validation_predictions.csv    验证预测与真实标签
    checkpoint_info.json          所用权重 SHA256、环境、配置及完成状态
    submission.csv                本次正式预测结果
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
相同配置的旧 state_dict 检查点仍可使用；移动目录不改变模型结构和权重键名。

## 训练方案与验证范围

configs/train.json 默认：320×320，batch=32，4 个读取进程，12 轮，AdamW，学习率 1e-4，混合精度。
图像等比例缩放补边，训练时 50% 水平翻转、±10° 旋转、轻微亮度/对比度增强；验证与预测不随机增强。
年龄空值、非有限值、<=0 或 >110 视为缺失，用训练数据中位数填补；均值和标准差仅从训练数据计算。
图像特征与年龄、缺失标记拼接分类，默认 BCE pos_weight=1，不同时叠加多种不平衡处理。
显存不足时在新实验配置中减小 batch_size；内存/数据加载问题可减小 num_workers。
pretrained=false 和非零样本 limit 只用于流程调试。配置改变请建新实验目录，不要混用检查点。
原图解码可能成为性能瓶颈，跑通后再按 GPU 利用率决定是否建立缩放缓存。

仅此前在 CPU 合成小样例上验证过训练/恢复流程；目录迁移后做轻量测试和路径检查，未在本地启动训练，未在 5090 实测。
数据顺序与增强按 seed、epoch、图片索引固定，跨 GPU/驱动/PyTorch 版本不保证逐位相同。
最终仍需官方 PyTorch 2.10.0/torchvision 0.25.0、T4 16GB 环境核验，单图端到端 <=100ms 和模型 <=500MB 尚未实测。
仅加载自己生成的可信权重。复赛按届时清单打包，不能把 last.pt 等整个 runs 目录直接当作提交模型包。

## 代码测试与版本控制

本地在 training 根目录执行以下轻量测试，不会训练模型：

```bash
python -B -m unittest discover -s tests -v
```

Git 仅位于 training/.git；数据、logs、runs、缓存均不入库。提交信息用 `关键词: 改动总结`，分支用 `关键词/用途`。
模型与私有数据不要公开上传，服务器不必携带 .git；需要 Git 记录诊断时再同步私有代码仓库。
