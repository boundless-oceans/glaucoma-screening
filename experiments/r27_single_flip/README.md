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

## 高细节局部微调试验入口（服务器真实检查待运行）

新增detail_training.py，独立项目、分阶段执行，不改变已经导出的候选。局部从固定R27 raw epoch12初始化，权重及官方train清单固定SHA核验；全图和定位器保持冻结，定位器关闭TF32，重新计算完整train原图、valid原图/翻转ROI，避免复用旧TF32资产。train51396/1657阳性、valid5849/191阳性，test不进入训练/验证。仅保存几何JSONL，不缓存全量512图片。

```bash
python experiments/r27_single_flip/detail_training.py prepare \
  --project /root/autodl-tmp/r27_detail512_pilot \
  --materials-root /root/autodl-tmp/r27_materials_20261102 \
  --data-root /root/autodl-tmp/glaucoma_first --max-hours 3
# 以下只能在空闲的Linux CUDA服务器上执行，沿用/root/r16-roi-env环境：
python experiments/r27_single_flip/detail_training.py assets --project /root/autodl-tmp/r27_detail512_pilot
python experiments/r27_single_flip/detail_training.py benchmark --project /root/autodl-tmp/r27_detail512_pilot
python experiments/r27_single_flip/detail_training.py smoke --project /root/autodl-tmp/r27_detail512_pilot
# 检查基准和预算后才执行：
python experiments/r27_single_flip/detail_training.py train --project /root/autodl-tmp/r27_detail512_pilot
python experiments/r27_single_flip/detail_training.py evaluate --project /root/autodl-tmp/r27_detail512_pilot
```

assets可能需要一段时间；三小时从assets第一次启动开始计时，包含后续阶段及手动等待时间，prepare不开始计时。模型阶段的--max-hours不会重置预算。预算检查在批次边界执行，不能保证计时瞬间强制退出。超时状态为paused、退出码3；已完成epoch的last和之前选中的best保留，半轮last标记partial_epoch，不参与选择。**此试验不支持恢复训练或自动续跑**，暂停后不能直接重跑同名阶段，避免覆盖证据。预算不足会拒绝启动训练；正式上机前必须核对剩余预算。

benchmark：相同固定880张train、每臂5批预热＋50批计时、batch16/4workers，三臂各重新加载相同父权重，执行真实前向/反向/AdamW；权重仅用于测量、不保存、不用于后续微调。报告有效AMP更新、显存峰值和每批耗时。只覆盖局部训练步骤，不能代替整轮或端到端延迟验收，缓存与执行次序会影响计时。可用新工程及--order detail512 low512 low384复测反序。

smoke：16阴性＋16阳性小样本训练，验证实际更新、checkpoint保存/读回与两类双视图推理；不进行模型选择。训练必须先通过benchmark和smoke；按新细节512耗时估计3轮不超过剩余预算70%，才会启动。这个估计不保证实际能跑完。

train：固定batch16、最多3轮，backbone lr3e-6、head lr3e-5、AdamW weight_decay1e-4，沿用阳性权重49739/1657以及原局部旋转/颜色增强。先完整评估同模型低解码384与高细节512第0轮，保留第0轮best；随后按融合后的翻转AUROC→AP选择最佳512权重，原图指标同时记录。每个完整epoch核验类别覆盖、必须有实际优化器更新，并保存last/optimizer/scaler；输入文件大小/mtime在后续阶段检查，不等价于所有JPEG逐文件密码学签名。

evaluate：最佳512局部用B1重新评估完整valid双视图，使用冻结的全图logit及ROI资产。它仍不是实时全流程延迟验收。没有自动test导出，也不能将新的候选权重直接塞入固定原R27推理入口；如果候选优于原版，还需另做实时定位/裁剪一致性、端到端延迟和导出适配验收。

本地3项入口CPU检查通过（准备无Torch、拒绝覆盖、完成标记篡改拒绝、Windows模型执行拒绝），4项裁剪检查回归通过。help/语法检查完成；**GPU资产生成、真实三路benchmark、smoke以及训练均尚未执行**，这些CPU检查不能代替服务器集成验收。运行前需打包新源码并在服务器新目录重新prepare，不覆盖旧候选目录。

2026-11-03服务器预检查：三种尺寸真实前向通过，官方train两类图片的优化器探针确认实际权重更新，checkpoint保存/严格读回及双视图验证前向通过；探针不是候选训练。首步AMP跳过不能直接判失败，smoke调整为初始scale8192、最多8次两批尝试，必须确认实际head变化；失败仍停止，不自动放宽有限性检查。此修复基于1a83c51，运行时源码哈希由新prepare固定，需另行提交。

已在/root/autodl-tmp/r27_detail512_pilot_20261103_v2启动assets，后台按assets→benchmark→smoke→train→evaluate执行，每阶段失败即停止。正式train仍受benchmark、smoke及剩余预算门限约束，不能把资产生成状态称为正在第几轮训练。三小时从11:27:34（北京时间）开始计时，约14:27:34耗尽预算；不自动关机。启动时尚未完成全量资产、三路基准或正式微调。日志为该项目logs/bootstrap.log；源码快照和AMP修复说明保存在项目目录。

### 高细节512三轮试验结果（2026-11-03）

资产、三路benchmark、smoke、三轮train及最终B1 evaluate全部完成，退出0；13:54:16结束，总计2小时26分42秒。9639次训练尝试、9631次成功更新，最佳为epoch3，父权重和第0轮记录保留。

|翻转结果|AUROC|AP|FN/FP @0.5|
|---|---:|---:|---|
|旧稳定384实时B1|0.9901099125|0.8737620979|14/147|
|本轮原权重低解码384 batch16|0.9901113005|0.8738171220|14/147|
|高细节512未微调epoch0 batch16|0.9892109398|0.8656975680|29/82|
|高细节512 epoch3最终B1|0.9902935935|0.8803755447|28/72|

最终原图AUROC0.9886571208，低于旧原图；此试验按预定翻转AUROC→AP选轮。翻转AUROC相对旧稳定实时B1增加约0.0184个百分点，幅度小；AP提高，但固定0.5漏检增多，不是全面改善，也不保证平台提分。

实测局部训练batch16每批low384/low512/detail512分别0.2427/0.2910/0.3394秒，CUDA峰值allocated分别约3.37/5.58/5.58GB；不是新候选实时端到端延迟。完整资产生成约一小时，也是本轮重要成本。

本地归档outputs/training_results/20261103_r27_detail512_3epoch，包含最佳/最后权重、资产、各轮预测、源码和日志。tar SHA256、64份备份文件及五阶段完成标记核验通过；CPU独立重算source384、epoch00/01/02/03、best_b1六份完整双视图valid指标一致，未在本地加载模型。最佳局部train/best.pt为111343455bytes，SHA256为3172d1858e4382019d5abb705aad618c7c5e6a0c06d123e74bfa103932e30c52。

当前尚未完成新局部权重的实时高细节推理、实时/缓存一致性及端到端延迟验收，没有生成该候选test提交。不得直接塞进固定原权重/384预处理的旧导出入口。下一步先验收新推理流程，再决定是否使用平台一次提交；旧稳定候选保留。

### 新权重实时验证入口与小样本结果

detail_adapter.py沿用同一全图/定位器和TF32关闭策略，实时定位后从原JPEG按该视图ROI读取512细节，加载固定epoch3局部权重。validate_detail_live.py固定核验best.pt SHA、三个候选预处理源码SHA、训练产物完成标记和B1参考；不训练或导出test。

独立服务器代码目录/root/autodl-tmp/r27_detail_live_code_20261103，输出目录/root/autodl-tmp/r27_detail_live_validation_20261103。check阶段固定37图双视图74次，核验ROI、解码/原图尺寸、裁剪RGB哈希、全图logit和最终概率；74次已通过，变化数均0，概率及全图logit最大差均0。evaluate自动继续完整5849张**翻转视图**，不是两视图全量，完成状态仍待确认。每图实时读取JPEG、运行全图和定位器、读取高细节ROI并运行局部模型，记录trace与延迟；不使用缓存分数作为实际预测。

小样本4090D平均157.36ms、p95217.80ms，73/74次超过100ms。计时含诊断裁剪哈希，四次预热不计入，模型加载单独记录；不是T4验收，当前流程存在明显延迟风险。正式部署尚未通过，不能因指标改善就称可提交部署。完整运行的指标及延迟待报告确认。上述文件是此次新增、尚未提交的验证源码，服务器训练快照未被修改。

2026-11-03收到服务器完整实时验证日志：5849张翻转预测通过，实时ROI和全图logit/概率均与B1缓存一致，最大概率差0；AUROC0.9902935935、AP0.8803755447。完整集平均133.43ms、p50为125.26ms、p95217.96ms、p99245.21ms，4083/5849张超过100ms。只有小样本74次显式比较了裁剪RGB哈希，完整集的crop_changes字段不代表逐图重新计算过缓存裁剪哈希。以上来自用户粘贴的完整成功日志，实时验证归档尚未下载；不冒充T4验收，仍未生成该候选test提交。

### 分段计时工具（不改变推理方案）

新增profile_detail.py和profile_detail_adapter.py。CPU模式只测共享解码/全图准备、完整JPEG解码/512裁剪、裁剪哈希；固定采样、两臂交替次序。CUDA模式每图对照普通/分段两次推理，段间显式同步会引入额外开销，因此原版未插桩总耗时单独记录，不能用分段耗时和宣称推理提速。细分全图、定位器、原图局部、局部模型和融合哈希；要求插桩前后ROI/裁剪哈希相同、概率差≤1e-7，并对照该候选B1参考。只诊断，不训练或导出。

```bash
python experiments/r27_single_flip/profile_detail.py cuda \
  --training-project /root/autodl-tmp/r27_detail512_pilot_20261103_v2 \
  --materials-root /root/autodl-tmp/r27_materials_20261102 \
  --data-root /root/autodl-tmp/glaucoma_first \
  --output /root/autodl-tmp/r27_detail_segment_profile \
  --samples 32 --repeats 3
```

test_detail_profile.py检查去掉计时语句后预测函数AST与实时预测表达式完全相同，已通过，不导入Torch。CPU模式本地32图×2重复已完成（私有outputs/training_results/20261103_detail_cpu_profile）：共享解码/全图准备平均163.42ms、完整解码/裁剪151.01ms、哈希3.50ms；Windows机器、NumPy版本和缓存情况不同，不能从此推断服务器耗时占比、速度收益或T4表现。CUDA分段计时仍待服务器空闲后运行。
