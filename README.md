# 青光眼分类训练代码

本目录上传到服务器即可使用。默认方案：ImageNet 预训练 ResNet18 + 年龄 + 年龄缺失标记。
仅训练 train，valid 只用于验证和选择权重，test 只用于预测。不需要新增人工标注。
本地现阶段用于编辑代码；不要直接在本机启动完整训练。下列正式训练命令在 AutoDL 终端执行。

## 1. 上传目录

```text
/root/autodl-tmp/glaucoma_first/
  train/                 原始图片
  valid/
  test/
  train.csv
  valid.csv
  test.csv
  training/              本目录的所有代码与配置
```

无需上传本地 work、outputs、Git 元数据或本地 Conda 环境。
代码按 CSV 名称不区分大小写匹配磁盘文件，因此兼容 CSV 的 .jpg 与磁盘的 .JPG。

数据集压缩包和 training 代码可以分开上传，先后顺序不限。压缩包在服务器解压后，
`--data-root` 指向直接包含 train/valid/test 和三个 CSV 的目录，不是压缩包路径，也不是多套一层的父目录。
training 不必放在数据目录里。例如数据放在 /root/autodl-tmp/glaucoma_first，代码放在 /root/autodl-tmp/training，
则在代码目录执行 `python preflight.py --data-root /root/autodl-tmp/glaucoma_first`，训练和预测也传同一个绝对数据路径。

## 2. 无卡模式先准备

进入服务器终端，选择以下一种环境准备方式。

### 方式 A：独立 Conda 环境

本目录的 enviroment.yml 是服务器专用：Python 3.12、torch 2.8.0+cu128、torchvision 0.23.0+cu128，
以及固定版本的 NumPy 和 Pillow。它与父目录的本地 cu126 配置不同。
无卡模式也可以安装，但必须等有卡开机后才能验证 GPU。

```bash
cd /root/autodl-tmp/glaucoma_first/training
conda env create -f enviroment.yml
conda activate glaucoma-cloud
python preflight.py --data-root ..
```

创建后无需再执行 requirements.txt 安装。以后每次新终端或切换到有卡实例后，先激活 `glaucoma-cloud`。
若 conda activate 提示 shell 未初始化，先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"` 再激活。
这不会修改镜像原有 base 环境，也不会安装系统显卡驱动。

### 方式 B：沿用镜像现有环境

```bash
cd /root/autodl-tmp/glaucoma_first/training
python -m pip install -r requirements.txt
python preflight.py --data-root ..
```

preflight.py 不加载 PyTorch、不使用 GPU、不训练，只检查 CSV、全部图片路径、两类标签和年龄参数。
上传后应看到 train=51396、valid=5849、test=5855；这里只核对路径，不能替代原始数据完整解码或上传文件哈希校验。

保留镜像自带的 GPU 版 PyTorch。AutoDL 镜像推荐配套为 Python 3.12、torch 2.8.0、torchvision 0.23.0、CUDA 12.8。
不要使用父目录的 enviroment.yml 覆盖云端环境：该文件是 Windows/4060 的 cu126 配置。
如果采用方式 A，则使用 training/enviroment.yml，不需要保留同一个 Python 环境中的镜像 PyTorch。

首次训练会下载官方 ResNet18 ImageNet 权重。为避免 GPU 空闲窗口浪费在下载上，可以提前运行：

```bash
python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.DEFAULT.get_state_dict(progress=True)"
```

此命令只下载权重，不训练、不调用 GPU。如下载失败，先解决网络，不能将 pretrained 改为 false 后当作正式预训练方案继续。

## 3. 有卡开机后检查并训练

```bash
cd /root/autodl-tmp/glaucoma_first/training
python gpu_check.py
python -u train.py --data-root .. --config config.smoke.json --run-dir runs/smoke
```

先跑小样本：256 张训练图、128 张验证图、16 张测试图、1 轮。成功后应生成 runs/smoke/DEBUG_predictions.csv，
只用于确认环境和流程，不能提交比赛，也不能用其指标判断最终效果。
这一条会实际使用服务器 GPU；本地不会自行启动训练。

确认小样本运行成功，再启动全量任务：

```bash
python -u train.py --data-root .. --run-dir runs/baseline --max-hours 3
```

`gpu_check.py` 才会执行一次很小的 GPU 前向/反向测试。无卡模式下报告 GPU 不可用是正常现象。

`--max-hours 3` 表示本次最多进行约 3 小时的训练/验证阶段，然后保存暂停；可按分配到的时长修改。
计时从模型、数据加载初始化之后开始，不含依赖安装、首次下载和结束后的测试集推理；验证阶段会完整结束，因此不是严格的实例关机定时器。
脚本不会关机、抢占 GPU、定时自动启动或申请平台资源。你需要在使用权到期前预留保存时间并主动停止/关机。

下一次执行相同命令，自动读取 runs/baseline/last.pt 接续。配置和 CSV 必须保持不变。
默认每 200 个训练批次、每轮训练结束前、每轮验证结束后保存一次；正常暂停会额外保存。
第一批训练前也会保存初始 last.pt，避免首次中断后仅留下配置文件而无法重新启动。
同一个运行目录由 .run.lock 文件的操作系统锁保护，锁文件保留是正常现象；进程退出会释放锁，不要手动删除锁文件来并发训练。
恢复训练时应一起保留 last.pt 和已有的 best.pt；最佳模型丢失会明确报错，避免训练结束后才发现无法预测。
每轮 12 个 epoch 是整个训练任务总量，续跑不会重新开始计数。

### 后台运行（可选）

```bash
mkdir -p logs
nohup python -u train.py --data-root .. --run-dir runs/baseline --max-hours 3 >> logs/baseline.log 2>&1 &
echo $!
```

记下打印出的进程 PID。查看进度：

```bash
tail -n 30 logs/baseline.log
```

需要让出 GPU 时，前台运行按一次 Ctrl+C；后台运行用 `kill -TERM 实际PID`。
等待日志出现保存暂停提示，确认进程退出后再关机。不要同时启动两个进程写同一个 run-dir。
强制关闭实例或 kill -9 无法触发保存，只能从最近一次已保存的检查点恢复。
后台日志路径特意放在 logs，不要在首次训练时把重定向日志放进 run-dir。

### 终端输出与日志的区别

- 直接执行 train.py：批次进度和每轮指标输出到终端；每 25 个训练批次和每轮最后一个批次打印一次。
- 无论是否重定向，每轮验证指标都会追加保存到 run-dir/history.jsonl；最佳指标还保存在 best_metrics.json。
- history.jsonl 不包含逐批次终端文本。要保存完整控制台输出，使用上面的 nohup 命令；`>>` 在续跑时追加，避免清空旧日志。
- 若想前台既看终端又保存日志，可运行：

```bash
mkdir -p logs
python -u train.py --data-root .. --run-dir runs/baseline --max-hours 3 2>&1 | tee -a logs/baseline.log
```

tee 是前台方式，不保证断开 SSH 后继续运行。需要断开连接时用 nohup 或自己的 tmux 会话。

## 4. 输出和提交

```text
runs/baseline/
  last.pt                  续训用，含优化器等状态，体积较大
  best.pt                  最佳验证模型，预测使用这个
  best_metrics.json        最佳验证指标
  valid_predictions.csv    最佳模型对验证集的预测
  valid_thresholds.csv     阈值 0.01～0.99 的验证表现
  history.jsonl            每轮验证日志
  age_stats.json           只用训练数据拟合的年龄参数
  config.json              本次配置快照
  environment.json         本次运行环境记录
  submission.csv           完整训练结束后自动生成
  submission.csv.json      预测来源与阈值记录，不需要上传比赛平台
```

若时间只够训练几轮，已有 best.pt 后，可以单独生成提交，无需等满 12 轮：

```bash
python predict.py --data-root .. --checkpoint runs/baseline/best.pt --output runs/baseline/submission.csv
```

暂停不会自动运行测试集推理，以免占用额外 GPU 时间。完整完成后默认自动预测，也可训练时加 `--skip-predict`。
正式数据集的 submission.csv 应有 5855 行数据，列名为 Image_name,Probability,Pred_Label，UTF-8，无额外索引。
脚本保持 test.csv 原始顺序和名字，拒绝重复名称及无效概率。

默认 Pred_Label 使用 0.5 阈值。FAQ 与官网对实际评分输入有冲突，应核实后再决定是否调阈值。
确认可以自选阈值后，才可在独立预测命令加入 `--threshold 0.3` 等参数；不能参考测试集反馈反复挑阈值。
Probability 始终为模型 sigmoid 输出，不按阈值改写。
AUROC 为主要选模指标，完全相等时比较梯形 AUPRC；同时记录 Average Precision，二者定义不同。
valid_thresholds.csv 另有 hard_label_auroc，便于理解按硬标签评估时的表现。不要把阈值搜索后的验证值当作独立泛化证明。

## 5. 默认训练配置及修改方式

config.json 默认：320×320、batch size 32、4 个读取进程、12 轮、AdamW、初始学习率 1e-4、混合精度。
全模型微调；轻度水平翻转、旋转和亮度/对比度增强；等比例缩放补黑边，不做随机裁剪。
年龄 <=0、>110、空值或非有限值作为缺失；中位数填补、均值和标准差只从训练数据计算。
图像特征与标准化年龄及缺失标记直接融合分类。训练损失默认普通 BCE，pos_weight=1，不自动把正负权重设成约 30 倍。

如显存不足，可在首次训练前减小 batch_size。进程过多或内存不足则降低 num_workers。
改变已有训练的配置会被拒绝，避免悄悄改变续训实验；新方案请复制配置并指定新的 --config 和 --run-dir。
不要移动、替换或修改一个正在续训任务的原始图片。CSV 校验并不能检查图片内容是否被替换。

train_limit/valid_limit/test_limit=0 表示全量。非零仅用于调试，标签采样会尽量保留两类。
调试运行仅输出 DEBUG_predictions.csv；独立 predict.py 拒绝使用带样本限制的检查点生成正式提交。
pretrained=false 仅适合离线流程测试，不是正式推荐配置。

## 6. 可复现与范围

检查点保存优化器、混合精度状态、随机状态、当前 epoch 和下一个批次位置。
数据顺序和增强根据 seed、epoch、图片索引确定。CPU 小样例验证了中途恢复与连续运行日志一致；跨 GPU、驱动或 PyTorch 版本不保证逐位相同。
恢复采用普通 state_dict，不依赖整模型 pickle；只加载本项目自己产生且可信的检查点。

训练仅在本机 CPU 临时合成数据上做了流程测试，未在本机训练真实竞赛数据，也未完成 5090 GPU 验证。
默认参数是起点，不保证分数。原始大图解码可能成为瓶颈，正式运行后根据日志和 GPU 利用率再决定是否建立缩放缓存。
比赛最终需要在官方 PyTorch 2.10.0/torchvision 0.25.0、T4 16GB 环境验证，单图端到端 <=100ms、模型总大小 <=500MB 还未实测。
best.pt 是模型参数检查点；last.pt 是续训产物。复赛应按届时官方清单打包，不要直接把整个 runs 目录当作提交模型包。
请把 best.pt、last.pt 和日志备份到自己的存储；数据和模型不要提交到公开 Git 仓库。

## 7. 本地 Git

Git 仓库仅建立在 training 目录内，只跟踪本目录的代码、配置和说明，忽略权重、运行日志和临时文件。
父目录 E:\glaucoma_first 不建立 Git 仓库，数据集、数据检查脚本和报告均在本仓库范围之外。
先进入 training 目录，再执行 Git 命令。
查看版本：`git log --oneline`；查看修改：`git diff`。
要取回旧版某个代码文件，先保存当前修改，再用 `git restore --source 提交号 -- 文件名`。
不必上传 .git 到服务器；如果通过 Git 同步代码，数据和权重仍需单独私密传输。
