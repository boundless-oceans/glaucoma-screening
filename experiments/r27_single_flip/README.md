# R27 单次水平翻转对照

将此前 R16 的单次翻转方法迁移到队长 R27，只有推理输入视图改变。固定全图 EfficientNet-B2/512 raw epoch7、监督定位器 R23 epoch50、局部 ConvNeXt-Tiny/384 raw epoch12，共约143.48MB权重。原图参考平台98.56、此前R16翻转98.55均为用户反馈，不是本实验的新结果。

翻转发生在原JPEG的draft768解码、EXIF校正和RGB转换之后，全图输入和定位器都读取翻转后的像素。定位器重新预测中心，再按原版source-field30规则生成局部区域：保留原图尺度、30%边长和越界黑色填充，固定50/50 logit融合。每张图只运行一次全图、一次定位器、一次局部模型；不是原图与翻转图平均。无年龄、无CAM、不复用原图ROI、不训练或修改权重。

## 本地检查

在training仓库根目录执行：

```bash
python experiments/r27_single_flip/test_cpu.py --materials-root /path/to/R27_extracted
```

不导入PyTorch、不读取权重状态、不使用本地GPU。8项合成协议检查、2项原版源码/合成图像检查；后两项需指定解压材料。源码检查以AST验证adapter与队长predict函数的差异严格限定为图像准备替换；图像检查覆盖EXIF方向、原图准备逐字节一致、镜像像素及越界填充。另有合成预测驱动prepare→check→evaluate→export及产物篡改拒绝的流程检查，不能替代服务器真实模型检查。

## 材料和工程准备

队长原压缩包解压到独立的私有目录，不覆盖现有训练目录。源码、config、三份权重、原版valid参考及官方valid/test清单均有固定SHA256；包含manifest本身，拒绝替换。真实逐行数据和权重留在仓库外，不提交Git。只使用官方valid/test图片，不重新分配训练数据。

`prepare`只核验文件、索引valid图片并保存绝对路径，不加载模型，可本地预检。服务器须重新prepare，不能直接上传Windows路径工程。工程输出必须位于仓库外或忽略的outputs/中。保留本实验目录和仓库src/glaucoma_training/data_utils.py。

```bash
python experiments/r27_single_flip/run.py prepare \
  --project /root/autodl-tmp/r27_single_flip \
  --materials-root /root/autodl-tmp/r27_materials \
  --data-root /root/autodl-tmp/glaucoma_first
```

## 服务器分阶段运行

Linux、Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128、CUDA。沿用/root/r16-roi-env，记录实际NumPy/Pillow和GPU版本。队长原环境未记录完整NumPy/Pillow版本，不能宣称跨机器逐位复现。

```bash
mkdir -p /root/autodl-tmp/r27_single_flip/logs
set -o pipefail
/root/r16-roi-env/bin/python -u experiments/r27_single_flip/run.py check \
  --project /root/autodl-tmp/r27_single_flip \
  2>&1 | tee /root/autodl-tmp/r27_single_flip/logs/check.log
# check完成后：
/root/r16-roi-env/bin/python -u experiments/r27_single_flip/run.py evaluate \
  --project /root/autodl-tmp/r27_single_flip \
  2>&1 | tee /root/autodl-tmp/r27_single_flip/logs/evaluate.log
```

check：固定16阴性＋16阳性，跑原版、翻转及关闭翻转的适配器审计。关闭翻转时要求ROI、裁剪RGB哈希等完全一致、概率差≤1e-7；原版相对队长CSV要求概率最大差≤0.001且0.5标签无变化。这个跨环境门限不是逐位一致承诺，失败时保留证据、单独调查，不自动放宽。

evaluate：5849张/191阳性完整valid，同一进程共享三份只读模型，逐样本交替原图/翻转执行次序。两臂各预热2次后计时，输出原版/翻转AUROC、AP、混淆矩阵、AUROC差值和原版参考审计。trace.jsonl保留每张图的logit、概率、定位中心、ROI、裁剪哈希与耗时，便于发现错误。小样本指标不与完整valid混比。计时只作当前GPU诊断，不能替代T4单图端到端≤100ms验收。

每阶段独占文件锁，新建目录且不覆盖已有结果；失败保留failed.txt及部分trace。所有文件写完且输入复核后才写complete.json，下阶段核验前阶段产物哈希。改源码或更新路径须prepare新工程；不自动重试、训练、上传平台或关机。

## 可选生成test候选

只在完整valid原版复现通过、翻转AUROC优于同机原版后手动执行。valid已被反复用于实验选择，改善不保证平台上涨；同时观察AP和实际耗时。不会自动搜索翻转/融合组合。

```bash
/root/r16-roi-env/bin/python -u experiments/r27_single_flip/run.py export \
  --project /root/autodl-tmp/r27_single_flip
```

输出5855行submission_single_flip.csv，官方顺序、连续概率17位有效数字、0.5硬标签及文件SHA256。测试无标签，不计算test AUROC。export同样保留逐图trace，不提交平台。

当前状态：10项CPU检查通过，队长材料哈希及本地prepare通过。Windows默认临时目录受沙箱权限影响，检查使用工作区outputs临时目录在解除该限制后运行；未导入PyTorch或使用GPU。尚未进行R27真实模型配对验证，暂无翻转收益结论。

## 复现失败后的离线排查

2026-11-02服务器完整配对已执行，但原版相对队长参考的最大概率差0.089172超出0.001门限，未生成完成标记或test提交。原版AUROC为0.989575526、AP为0.871035833；翻转AUROC为0.990119629、AP为0.873973351。当前只能报告这次同机对照，不能宣称原版复现验收通过。31张ROI不同：28张最多差1像素、3张有明显纵向偏移；概率超限3张与明显ROI偏移3张合计涉及5张图片。已重复检查的3张概率异常样本同机结果稳定，跨环境具体原因仍未知。

独立CPU诊断工具不修改原版模型、项目完成状态或运行源码：

```bash
python experiments/r27_single_flip/tools/audit_preprocessing.py \
  --materials-root /path/to/R27_extracted \
  --data-root /path/to/glaucoma_first \
  --output /path/to/new_preprocessing_fingerprints.json
```

固定核查valid_00163、02776、03407、05455、05673五张图，记录JPEG、EXIF/RGB解码、全图输入、定位器RGB256及归一化float32数组哈希和相关源码/依赖版本。输出必须是新文件；不导入PyTorch、不加载权重。本地五图指纹生成已通过，但不能代替队长原环境运行。两边按上述顺序比较：首个不一致层决定先查数据、解码/缩放还是归一化；若所有输入均一致，再在服务器比较定位热图及计算精度设置。热图存在接近峰值仅解释敏感机制，不能单凭它认定差异原因。

### 定位器TF32五图对照（服务器诊断完成，暂不采用）

实际4090D的五图预处理指纹已与队长完全一致；同一队长脚本在4090D复现了定位热图差异。相关输出归档在本地忽略的outputs/training_results/r27_gpu_replay_20261102。两边记录的cuDNN版本与TF32开关相同，具体差异不能唯一归因于某一个环境因素。

下一步只比较五图原版与定位器关闭TF32，各重复3次，交替次序。模型、图像、分类器精度、ROI规则及阈值不变。独立脚本临时包装定位器forward，在内部cuDNN上下文关闭TF32，调用结束和异常退出后恢复；不覆盖原版源码或先前阶段完成标记。热图转CPU用于诊断，其耗时不能用作推理验收。没有训练、选模或导出提交。

```bash
/root/r16-roi-env/bin/python -u \
  /path/to/training/experiments/r27_single_flip/tools/audit_locator_tf32.py \
  --code-root /root/autodl-tmp/r27_single_flip_code_20261102/training \
  --materials-root /root/autodl-tmp/r27_materials_20261102 \
  --data-root /root/autodl-tmp/glaucoma_first \
  --output-dir /root/autodl-tmp/r27_locator_tf32_audit_20261102
```

本地help和Windows拒绝模型执行入口检查通过，未加载模型。服务器完成五图两种设置各3次，共30次推理；独立核验归档/完成标记哈希及每臂重复热图、ROI、概率完全一致，全图分类logit在两臂相同。关闭定位器TF32使02776、05455的ROI恢复到历史框，但00163、03407、05673的ROI及概率未变，三个概率超限仍未解决。故暂不采用，不启动完整验证或生成test提交。产物归档在本地忽略的outputs/training_results/r27_locator_tf32_audit_20261102。

仅同一台4090D开启/关闭TF32的比较不能证明跨GPU一致性，需要与队长运行同一诊断后的热图及ROI继续核对。关闭TF32也不保证与历史原版结果一致，不通过反复调参追求三张历史概率。

跨机报告比较工具为tools/compare_tf32_reports.py，仅用CPU。它要求两份报告各含完整五图、两臂、各3次且每臂ROI/中心/概率/峰值索引重复一致；比较每臂跨机ROI、中心和概率。自比较五图全部一致、重复记录和重复不一致拒绝检查已通过，没有加载模型。完整交付包在忽略的outputs/training_results/r27_tf32_team_handoff_ready_20261102.zip，包含core.py、固定材料清单、诊断/比较脚本、4090D参考和run_captain.sh。此前小包未包含core.py及材料清单，应使用此次完整包。队长的同设置TF32对照尚未收到，不能预报结果。

### 2026-11-03跨GPU回复：支持进入新候选完整验证

队长同脚本结果已收到，完成标记哈希与热图独立核验通过，重新生成的比较与其comparison.json一致。原版定位器TF32开启时，五图两台机器ROI均不同；两边均关闭定位器TF32后，5/5完整定位热图逐元素相同、中心和ROI全部一致。最终概率最大差约0.00001586783，FP16分类器并非逐位一致。每臂三次重复稳定。

因此修正上述仅有单机诊断时的暂缓结论：值得将“仅定位器关闭TF32”作为独立候选进行完整valid和延迟验证；不代表已在所有图片/其他GPU通过，也不保证平台提分。结果归档在忽略的outputs/training_results/20261103_r27_crossgpu_tf32。

新候选必须独立建工程：原图和翻转都采用同一关闭定位器TF32策略；明确旧参考采用不同策略，保留其差异审计，不冒充复现旧概率。不得修改旧失败阶段的完成标记或放宽旧门限。当前run.py仍为原版设置；新设置使用下方独立入口。

## 独立稳定性候选入口（代码已检查，服务器完整验证待运行）

`run_stable.py`固定采用仅定位器TF32关闭策略，不提供其他精度/阈值搜索选项。`stable_precision.py`将该上下文限定在定位器forward内，包括异常路径的恢复，分类器仍使用队长原版FP16设置。权重及上游源码哈希保护不变；额外固定队长TF32诊断report.json的SHA256（fc14c192...），拒绝替换为任意参考。

```bash
python experiments/r27_single_flip/run_stable.py prepare \
  --project /root/autodl-tmp/r27_stable_candidate_20261103 \
  --materials-root /root/autodl-tmp/r27_materials_20261102 \
  --data-root /root/autodl-tmp/glaucoma_first \
  --tf32-reference /path/to/captain_tf32_reply/captain_result/report.json

python experiments/r27_single_flip/run_stable.py check \
  --project /root/autodl-tmp/r27_stable_candidate_20261103
python experiments/r27_single_flip/run_stable.py evaluate \
  --project /root/autodl-tmp/r27_stable_candidate_20261103
```

必须在新代码副本和新工程中prepare，不覆盖旧服务器代码。先check，成功才evaluate；不自动导出test。原来的固定32张抽样增加五张异常图，去重后实际37张；两臂重复概率差≤1e-7、裁剪RGB/ROI等逐值一致，关闭翻转适配器与原图实现也须一致。五张原图相对队长**TF32关闭**参考ROI必须完全相同且概率差≤0.001。check全量trace纳入完成标记哈希。

完整5849张对照中，check的原图/翻转样本必须全部与当前新设置的小样本记录一致；再检查队长五张参考。报告中的baseline_gate明确列出这些通过条件，旧TF32开启的reference_audit只作差异记录并注明设置不同。它不是对旧失败阶段的门限放宽，且不声称所有图片跨GPU相同。完整指标及当前GPU延迟均重新计算；T4仍待验收。显式export还要求新设置完整验证通过且翻转AUROC优于同设置原图。

本地`test_stable.py`两项检查通过，覆盖正常/异常时定位精度上下文恢复，以及合成prepare→check→evaluate→export流程、旧参考不一致但新条件明确通过、重复不一致和五图ROI异常拒绝；原10项CPU检查回归通过。没有本地模型加载或GPU运行。服务器新入口的真实小样本集成和完整验证仍待进行，不能把合成检查当作真实模型结果。

### 2026-11-03稳定候选完整运行与归档

提交b7cb7f9在4090D独立目录完成37图真实check、5849张配对evaluate和5855张翻转export，均正常退出。完整验证重放74条check双臂记录一致，五张同策略跨GPU参考ROI相同，概率最大差0.00001586783。新baseline_gate通过；旧TF32开启参考的差异仍如实保留。

|指标|同设置原图|单次翻转|
|---|---:|---:|
|AUROC|0.9895796898|0.9901099125|
|AP|0.8714801937|0.8737620979|
|FN/FP，阈值0.5|13/141|14/147|
|4090D平均端到端ms|76.81|79.04|
|4090D p95 ms|110.74|117.65|

完整验证耗时930.47秒；test导出474.48秒，预测阳性375。未训练、未改权重、未提交平台；T4和单图100ms仍未验收，平台提分未知。

本地私有归档位于outputs/training_results/20261103_r27_stable_single_flip，包含源码快照、固定参考、报告、逐图trace及日志，不含权重。核验远端tar SHA256和44份备份文件哈希，CPU独立重算完整valid指标一致，并核验5855行test顺序/概率/标签/trace。platform_submission/submission.csv是原始导出的字节复制，SHA256为a5849ab71cfbc1baa12764740310bbcc7d4a4d43d53d1568ed65dac761941ab7；不得将日志中的本地指标当作平台分数。所有归档保留在Git忽略目录，不提交逐行数据。

### 局部分辨率审计与下一候选准备

检查队长51396张train的固定ROI资产，解码后裁剪短边中位数356像素，41198张短边不足384，49541张不足512（约96.4%）；原图对应正方形区域边长中位数738。新稳定候选valid的裁剪短边中位数同为356，5631/5849张不足512。此项只检查几何尺寸，不使用疾病标签拟合。结果在忽略的outputs/training_results/20261103_r27_resolution_audit。

因此仅将现有低分辨率ROI放大至512并不保证增加可用细节。准备的detail_crop.py原型保留固定定位ROI的物理范围，将整数解码坐标按EXIF校正后的原图尺寸映射回原JPEG，读取原图像素再生成512输入，保留越界黑填充；定位器和全图仍可沿用现有共享解码。源坐标映射本身有整数取整，不宣称与旧裁剪像素相同。全分辨率JPEG解码会增加CPU/内存成本，真实吞吐、显存、延迟与指标仍需测量。

test_detail_crop.py两项合成CPU检查覆盖物理范围映射、越界填充、EXIF旋转和尺寸不匹配拒绝，另对5张官方train JPEG真实预处理检查通过；不导入PyTorch。该原型**未接入训练、验证或导出**，不改变当前已归档候选。后续如进行512短程微调，须在新工程中重新生成同设置ROI、固定全图和定位器，仅训练局部；训练/验证均采用相同的高细节输入流程，并保留epoch0候选。还应先测小样本吞吐，确认3小时预算可行，再承诺轮数；不能直接复用当前只认固定原局部权重的导出器加载新权重。

同日进一步复核：原型支持显式mirrored视图；调用方必须传入**该视图重新定位后**的ROI，不能把原图ROI直接作为翻转ROI。实现先将翻转视图框反射回原图坐标，裁剪后只镜像局部区域；与整图镜像后裁剪相同，包括越界黑填充，避免额外的整图镜像复制。EXIF已校正且为RGB时不进行整图exif_transpose/convert复制。真实解码仍会占用CPU和内存，不宣称已证明训练提速。

现4项合成CPU检查通过，新增非法/非有限尺寸拒绝与含越界的翻转等价性；另对5张官方训练JPEG各运行原图/翻转两种预处理，共10次均通过。上述检查不运行定位器，使用固定几何仅检验裁剪函数；并未验证翻转定位效果。未使用本地GPU。后续试验仍须先比较“低解码384、低解码放大512、原图细节512”的输入/吞吐，再决定训练，不能将放大带来的变化解释为细节收益。
