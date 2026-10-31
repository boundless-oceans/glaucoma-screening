# 原版 R16 水平翻转 TTA 对照

只做一项固定推理对照，不重新训练。使用队长原版 R16 的 E32 第7轮全图和 E33 第15轮局部权重；不能用 ROI 一致性实验新训练的局部权重代替。

## 固定方案与当前状态

- 原图臂直接调用未修改的历史 `R08Pipeline.predict_details`。
- 翻转臂在原始 JPEG 的 draft768 解码、EXIF 转正和 RGB 转换后水平翻转，不重新压缩 JPEG；随后重新去黑边、补边、计算全图 CAM、定位并裁取局部。
- 每个视图内部沿用原版全图/局部 raw-logit 等权融合；该视图定位失败则只用该视图全图。两个视图之间固定平均**概率**，阈值仍为0.5。
- 串行 B1、FP16、无年龄，不搜索视图权重、阈值或其他翻转组合。
- 当前通过19项本地 CPU 检查，增加了独立配对基线的绑定、防篡改及环境/推理代码一致性检查。没有在本地导入 PyTorch。
- 2026-10-30已运行原版32图真实检查，全部通过；完整TTA验证保存3117行后，因 `valid_03118.jpg` 原图概率0.0475584023与归档0.0579713952不一致而停止。该次严格归档对照未完成；后续独立同机配对的完整TTA已完成，见下一节。
- 额外诊断中，该图当前CAM的两个邻近峰值约4.493500/4.493049，ROI与旧资产不同；仅诊断使用旧框时概率约0.0579713952。支持选峰敏感性解释，但未确定跨环境差异来源，不能据此声称数据损坏或直接改变选峰规则。
- 完整baseline-audit已在4090D完成：5849张仅1张超逐图容差，全部0.5硬标签一致；本机AUROC=0.9893756512、AP=0.8644753091，归档为0.9893770392与0.8645259663。严格复现状态仍为false，不修改该记录。
- **新增paired-check/paired-valid模式改用明确绑定的本机审计概率作实验对照，历史归档仅作为背景参考；原check/valid保护不变。paired-check已在服务器通过，完整paired-valid也已完成；不能承诺平台收益。**

本实验复用相邻 `../r16_roi_consistency/vendor` 的固定历史源码，以及 `../r16_roi_consistency/scripts/manifest_data.py` 的本地清单生成器。上传时保留两个完整相邻实验目录（包含scripts），不能仅上传本目录与vendor。仓库不附带真实清单；prepare从本地官方valid.csv生成与历史SHA256逐字节一致的valid_manifest.csv，继续与外部参考预测及旧审计绑定。


## 2026-10-30完整配对结果

5849张本次原图概率与冻结本机审计基线逐张完全一致。原图AUROC=0.9893756512、AP=0.8644753091；等权水平翻转TTA为0.9899322462与0.8655545806。AUROC增加约0.00055660（0.05566个百分点）；0.5阈值FN从13降至10，FP从186降至171。

2000次按类别分层的图像级配对bootstrap，seed20261030，AUROC增量95%区间约[-0.000315,0.001718]，包含0。valid经过多次选模研究，且没有患者级独立性验证，不能据此保证平台收益。单独翻转臂仅为诊断，不事后搜索其他组合。

同轮4090D原图均值61.83ms/p95 88.08ms，TTA均值125.12ms/p95 179.48ms，代价约翻倍；未通过T4单张100ms验收。TTA保留为候选，暂不默认替换原版；未生成test提交，也未重训。工作区outputs/training_results/20261030_r16_hflip_tta_paired/归档45份校验文件，含实际源码、完整预测、计时、bootstrap和日志。本地纯CPU独立重算全部5组指标及逐图概率平均一致。

## 单次水平翻转候选：新增入口

此前双视图诊断中，单独翻转臂AUROC=0.9901034351、AP=0.8600360104；原图为0.9893756512与0.8644753091。它的AUROC较高、AP较低，属于看过valid后挑出的新候选，不能称为预先注册的盲测提升。新增入口没有改变翻转、CAM或ROI算法；只在部署路径中去掉原图臂和概率平均。19项CPU契约检查通过；2026-10-30已完成服务器32图single-check、5849图single-valid及5855图single-test。独立valid AUROC=0.9901034351、AP=0.8600360104，逐图概率与先前翻转臂完全相同。valid均值58.53ms/p95 84.62ms，test均值60.85ms/p95 88.16ms，均为4090D而非T4验收。

准备全新运行目录后依次执行；不能覆盖已有TTA项目或继承旧源码的通过标记：

```bash
# 新项目准备仍用scripts/prepare.py；此处示意该项目内的命令
python -u run.py --mode single-check --baseline-audit /path/to/completed/baseline-audit_TIMESTAMP
python -u run.py --mode single-valid --baseline-audit /path/to/completed/baseline-audit_TIMESTAMP
# 独立完整valid成功后才可显式导出test候选
python -u run.py --mode single-test
```

single-check固定32张，额外运行原图、翻转和关闭翻转helper来核对工程等价性，写独立single_check_passed.json；其耗时不能代表部署速度。single-valid只做单次翻转推理，报告单独时延和完整valid指标，并与冻结原图基线及历史归档比较。它复用原本已通过的审计材料校验规则，不更新模型、年龄统计或ROI。

single-test要求同一新项目、源码和实际运行环境下已完成5849图single-valid；准备时若存在test.csv，会绑定其SHA256，没有绑定则拒绝导出，需重新prepare。test元数据必须为5855个唯一、无标签的官方图像身份，输出严格保留test.csv顺序，兼容磁盘.JPG与元数据.jpg。仅生成submission_single_flip.csv，列为Image_name、Probability、Pred_Label，阈值0.5。概率使用17位有效数字并回读校验，避免靠近0.5时因写出舍入而出现概率与标签不一致；写入临时文件，完整核验后才改为最终文件名。

新出口仅通过显式single-test启用，旧check/valid、paired和审计入口都不导出test；不连接比赛平台或自动提交。测试集没有标签，不计算测试AUROC。单次推理虽减少一次全图/局部预测，实际时延仍须测量，不能据此宣称通过T4 100ms。

候选已归档至工作区outputs/training_results/20261030_r16_single_flip/submission_single_flip.csv，5855行、411个预测阳性；顺序、概率精度和阈值标签完整核验通过，58份导出文件哈希核验通过。本地纯CPU独立重算指标与服务器一致。没有自动提交平台。AUROC增量的2000次图像级分层配对bootstrap 95%区间约[-0.000716,0.002738]，包含0；该候选AUROC较高而AP下降，不能保证平台涨分。保留原版作对照。

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

复制推理源码，从本地官方CSV生成valid角色清单，并复制用户提供的已归档验证概率，生成 `EXPERIMENT.json`；权重保持外部只读。验证官方 valid 的5849个身份及标签与原资料一致，其中191例阳性。不读取 train/test 图像，不创建训练任务或安装环境。

## 3. 完整原版基线审计（已完成流程）

新源码必须重新prepare到全新运行目录，保留旧失败现场，不能覆盖旧项目或修改旧哈希记录。下方示例假定新目录仍命名为 `/root/autodl-tmp/r16_hflip_tta`；若该路径已经存在，准备命令和后续cd都换成同一个新路径。

```bash
cd /root/autodl-tmp/r16_hflip_tta
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
nohup /root/r16-roi-env/bin/python -u run.py --mode baseline-audit \
  > logs/baseline_audit.log 2>&1 < /dev/null &
```

只调用原版R08Pipeline跑完整5849图，不执行翻转或TTA，不更新权重和ROI规则。记录每张归档差异；超过0.001的样本继续收集，不在首个异常处终止。为每个异常额外保存JSON：重复原版预测、不翻转helper对照、JPEG与512输入像素哈希、几何、ROI和CAM前两峰差值。额外CAM来自另一次前向，明确不是计时调用的内部捕获；这些诊断前向不计入单图原版推理时延。近峰差值只描述敏感性，不是新选峰阈值。

审计完整跑完时 `status=completed`，但 `results.json/reproduction/baseline_reproduced` 单独表示是否满足原来的逐图及AUROC/AP容差；两者不能混用。任何逐图超限都会使复现判定为false。审计不写check_passed标记，也不自动启动TTA。若已有审计未完成、运行环境不同或复现失败，valid模式会在推理前拒绝继续；需查明原因后再决策，不默默放宽容差。

报告保存在 `outputs/baseline-audit_<UTC时间>/`，异常详情在其中的 `outliers/`。超时仍按失败处理并保留部分输出。没有历史原图哈希可用于反向验证时，不宣称JPEG内容与队长机器已经逐字节核验。

## 4. 基线问题解决后：32图真实检查

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

## 5. 再运行完整 valid 对照

```bash
nohup /root/r16-roi-env/bin/python -u run.py --mode valid \
  > logs/valid.log 2>&1 < /dev/null &
```

默认推理循环最多30分钟，超时保留部分输出并标记失败，不自动重试。实际耗时待测。可用 `tail -f logs/valid.log` 查看每25张的进度。

原图、翻转、TTA 及归档参考在同一5849图上对齐，报告 AUROC、AP、梯形 AUPRC、固定0.5的混淆矩阵、回退数、原图→TTA判别变化和指标差值。完整原图 AUROC 与参考绝对差需不超过0.0001，AP差不超过0.0005；这属于复现保护，不是涨分目标。失败时先核对材料、数据和环境，不直接放宽保护。

原图与两视图总耗时分别给出均值、p50、p95和最大值，包含读图、解码、预处理、实时 CAM/ROI、前向及 GPU 同步；额外的不翻转等价检查不计入耗时。前三图预热不计入，视图顺序固定、操作系统缓存不受控，因此不是严格缓存对照。4090D结果不能作为T4单图100ms验收；TTA是否适合最终部署仍要看目标硬件实测。

## 输出与后续决策

### 完整审计后的独立同机配对对照

跨环境归档有1张逐图超差，但整体指标差异很小；本次采用新的、明确记录的配对协议，不把旧复现失败改写成通过，不针对该图指定旧ROI。准备新的运行项目后，显式传入已完成审计的输出目录：

```bash
python -u run.py --mode paired-check --baseline-audit /path/to/completed/baseline-audit_TIMESTAMP
python -u run.py --mode paired-valid --baseline-audit /path/to/completed/baseline-audit_TIMESTAMP
```

入口验证审计complete哈希、5849图身份与标签、参考概率、权重、官方CSV、推理源码、实际运行环境，以及原图整体AUROC/AP仍在原容差内。所有归档超差图必须有重复推理一致、不翻转helper一致的诊断。新项目可复用旧审计，但模型构建、预处理和推理源码不得变化。32图paired-check写独立标记，完整paired-valid必须使用同一源码/环境/审计；不写原check_passed或修改历史严格复现判定。

每张新原图概率与冻结本机基线比较，仍使用0.001逐图容差，最终AUROC/AP仍按原容差核对；这是新的对照对象，非放宽历史复现阈值。predictions.csv同时保留archived_probability（队长归档）和reference_probability（本机审计），results.json的paired_control记录来源与历史严格复现状态。TTA始终与同一机器新测原图配对比较，不能把历史差异计为TTA收益。不训练，不改CAM/ROI规则，不生成test提交。

每次写入全新的 `outputs/<mode>_<UTC时间>/`；single-test额外包含submission_single_flip.csv：

```text
outputs/valid_<UTC时间>/
├── registration.json       # 材料、源码、配方、环境和数据角色
├── predictions.csv        # 原图、翻转、TTA、参考概率与逐图耗时
├── results.json            # 对照指标、复现差异和时延
├── status.json             # running/completed/failed/interrupted
└── complete.json           # 成功后最后写入的报告哈希
```

不覆盖旧目录，只有显式single-test生成test候选文件；各模式均不操作比赛平台。失败输出可能只有部分文件，不能当作完整验证。当前 valid 已用于多次选模和研究，即使 AUROC 上升，也不能据此宣称获得独立盲测提升。先同时评估 AUROC、AP 与目标硬件耗时，再决定是否保留这一个固定方案。
