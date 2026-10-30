# 原版 R16 水平翻转 TTA 对照

只做一项固定推理对照，不重新训练。使用队长原版 R16 的 E32 第7轮全图和 E33 第15轮局部权重；不能用 ROI 一致性实验新训练的局部权重代替。

## 固定方案与当前状态

- 原图臂直接调用未修改的历史 `R08Pipeline.predict_details`。
- 翻转臂在原始 JPEG 的 draft768 解码、EXIF 转正和 RGB 转换后水平翻转，不重新压缩 JPEG；随后重新去黑边、补边、计算全图 CAM、定位并裁取局部。
- 每个视图内部沿用原版全图/局部 raw-logit 等权融合；该视图定位失败则只用该视图全图。两个视图之间固定平均**概率**，阈值仍为0.5。
- 串行 B1、FP16、无年龄，不搜索视图权重、阈值或其他翻转组合。
- 当前通过7项本地 CPU 检查，覆盖身份/标签对齐、旧权重限制、概率平均、EXIF与奇数尺寸下的原图像素等价、翻转解码、CLI及准备保护。没有在本地导入 PyTorch。
- **真实模型的32图检查、完整验证和 GPU 耗时测量尚未运行。没有证据表明 TTA 会提高平台得分。**

本实验复用相邻 `../r16_roi_consistency/vendor` 中固定的历史源码，准备时检查其清单。上传配方时需要保留这两个相邻目录，不能只上传本目录。

## 所需材料

| 材料 | SHA256 |
| --- | --- |
| 原版全图 E32 epoch7 | `b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2` |
| 原版局部 E33 epoch15 | `1eb7f8627be0c341ed1ac426e8d4ce463475134edca8542cc201679841914175` |
| 原版验证概率 `valid_predictions.csv` | `40aa3754c94895aaf0f4d67e60a1f8eab206e2cd7562017d3b2ec792c9c05296` |

验证概率来自 `R16_9848_team_complementarity_20261029` 资料包，不是本次 ROI 一致性实验的预测。2026-10-30已收到并核验 `R16_E33_epoch15_original_local_20261030.zip`，其中9个登记文件均通过大小及SHA256检查，局部权重与上表一致，模型构建源码也与已固定的vendor一致。现在原全图、原局部及验证概率均已具备；这仅是文件核验，没有加载模型。

本地原局部权重位于工作区 `outputs/r16_original_local_20261030/source/runs/E33-r16-official-local-20261026-235935/best.pt`，核验记录为该材料目录下的 `verification.json`。准备过程只读取权重文件计算哈希，不反序列化、复制或下载模型。

运行环境复用服务器已经准备好的 `/root/r16-roi-env`：Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128，以及对应 NumPy、pandas、Pillow、scikit-learn。模型入口会检查 PyTorch/torchvision 版本，并在检查标记中记录实际完整运行环境。不要用旧的 glaucoma-cloud 2.8 环境运行。

## 1. 本地 CPU 检查

在本目录执行（需要 NumPy、Pillow）：

```bash
python tests/test_tta.py
```

测试中的权重是不可反序列化的假文件，仅检查准备流程；这些测试不代表真实模型验收。

## 2. 在服务器准备新目录

从本配方目录执行，替换三个材料路径；输出目录必须不存在：

```bash
/root/r16-roi-env/bin/python scripts/prepare.py \
  --output /root/autodl-tmp/r16_hflip_tta \
  --data-root /root/autodl-tmp/glaucoma_first \
  --global-checkpoint /path/to/original_E32_epoch7_best.pt \
  --local-checkpoint /path/to/original_E33_epoch15_best.pt \
  --reference-predictions /path/to/original_R16/valid_predictions.csv
```

只复制推理源码、官方 valid 角色清单及已归档验证概率，生成 `EXPERIMENT.json`；权重保持外部只读。验证官方 valid 的5849个身份及标签与原资料一致，其中191例阳性。不读取 train/test 图像，不创建训练任务或安装环境。

## 3. 先运行32图真实检查

```bash
cd /root/autodl-tmp/r16_hflip_tta
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
/root/r16-roi-env/bin/python -u run.py --mode check > logs/check.log 2>&1
```

固定种子20261030，从 valid 抽取16阳性及16阴性，仅做工程核对，不把这个平衡小样本的指标当完整验证成绩。每图检查：

1. 原图概率与队长归档概率的绝对差不超过0.001；超过则中止并保留部分报告。
2. 新 helper 关闭翻转时的概率、两分支 logit 和回退结果，必须与未修改的原版路径完全一致。
3. 翻转臂的概率必须有限且位于 `[0,1]`。

通过后写出 `check_passed.json`，绑定当前源码、两份权重、官方 CSV、归档概率和实际运行环境。再次运行检查会先撤销旧标记；完整验证必须有同一项目和环境下的成功检查。此处不更新任何模型参数。

## 4. 再运行完整 valid 对照

```bash
nohup /root/r16-roi-env/bin/python -u run.py --mode valid \
  > logs/valid.log 2>&1 < /dev/null &
```

默认推理循环最多30分钟，超时保留部分输出并标记失败，不自动重试。实际耗时待测。可用 `tail -f logs/valid.log` 查看每25张的进度。

原图、翻转、TTA 及归档参考在同一5849图上对齐，报告 AUROC、AP、梯形 AUPRC、固定0.5的混淆矩阵、回退数、原图→TTA判别变化和指标差值。完整原图 AUROC 与参考绝对差需不超过0.0001，AP差不超过0.0005；这属于复现保护，不是涨分目标。失败时先核对材料、数据和环境，不直接放宽保护。

原图与两视图总耗时分别给出均值、p50、p95和最大值，包含读图、解码、预处理、实时 CAM/ROI、前向及 GPU 同步；额外的不翻转等价检查不计入耗时。前三图预热不计入，视图顺序固定、操作系统缓存不受控，因此不是严格缓存对照。4090D结果不能作为T4单图100ms验收；TTA是否适合最终部署仍要看目标硬件实测。

## 输出与后续决策

每次写入全新的 `outputs/check_<UTC时间>/` 或 `outputs/valid_<UTC时间>/`：

```text
outputs/valid_<UTC时间>/
├── registration.json       # 材料、源码、配方、环境和数据角色
├── predictions.csv        # 原图、翻转、TTA、参考概率与逐图耗时
├── results.json            # 对照指标、复现差异和时延
├── status.json             # running/completed/failed/interrupted
└── complete.json           # 成功后最后写入的报告哈希
```

不覆盖旧目录，不生成 test 提交文件，也不操作比赛平台。失败输出可能只有部分文件，不能当作完整验证。当前 valid 已用于多次选模和研究，即使 AUROC 上升，也不能据此宣称获得独立盲测提升。先同时评估 AUROC、AP 与目标硬件耗时，再决定是否保留这一个固定方案。
