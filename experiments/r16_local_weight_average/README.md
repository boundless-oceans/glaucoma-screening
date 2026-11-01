# R16固定局部权重平均

独立实验，只将原版E33 epoch15与新EMA epoch11局部ConvNeXt-Tiny/384的参数固定各取50%，全图E32 epoch7、实时CAM/ROI和视图内50/50 raw-logit融合保持原版。只生成一份新局部网络，不重训、不搜索系数、不叠加翻转、不自动导出test或提交平台。现有r16_inference的原版权重哈希保护不修改。

已有CSV的一次固定概率平均诊断AUROC为0.989756、AP为0.865632，仅提示有限互补。**概率平均不是参数平均，不能据此预测本实验分数。** [Model soups（ICML2022）](https://arxiv.org/abs/2203.05482)提供共享预训练初始化模型权重平均的研究依据，不保证本任务收益。官方valid已反复研究，不是盲测。

## 范围与来源

- 全图固定SHA：b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2。
- 原局部固定SHA：1eb7f8627be0c341ed1ac426e8d4ce463475134edca8542cc201679841914175。
- 新EMA固定SHA：98841189b6d165360cbeb93eb96f7c15dea8f42784ac925251817f60945d33dd。
- 两个局部config必须都是ConvNeXt-Tiny/384、纯视觉、seed20261026、相同通用ImageNet初始化和同一冻结全图。完整state_dict键/shape/dtype必须一致且有限；参数必须FP32。使用实际模型named_parameters识别参数，所有非参数状态必须相同才复制，不能平均整数或不同buffer。
- build生成新目录，父文件只读，执行前后核对输入哈希；输出去除优化器/训练恢复状态，不伪称某个epoch的新训练结果。
- 只评估官方valid（5849行、191阳性）与已完成的原版/EMA参考输出；不使用年龄，不读取test，不拟合任何数据。

## 本地CPU检查

在仓库根目录执行；不导入PyTorch、不加载权重或使用本地GPU：

```bash
python experiments/r16_local_weight_average/tests/test_average.py
```

8项本地检查通过，覆盖固定平均算术、buffer/结构/配置拒绝、参考顺序/概率保护、固定抽样、完成标记、来源篡改拒绝和纯文件打包；不能代替服务器真实模型验收。Windows沙箱可能拦截最终目录rename，本次纯CPU检查在相同授权工作区解除该限制后通过，没有加载模型。服务器build/check/evaluate已完成，实际结果见文末。

## 准备独立工程

prepare仅做复制和文件哈希，可在本地检查来源，但Windows绝对路径不能直接上传作为服务器工程。正式准备在Linux服务器上执行，保留相邻r16_roi_consistency与r16_inference源码目录、仓库src/glaucoma_training/data_utils.py。无需r16_local_ema训练代码，已完成的EMA权重和参考CSV作为外部只读材料。

```bash
python experiments/r16_local_weight_average/scripts/prepare.py \
  --output /root/autodl-tmp/r16_local_weight_average \
  --data-root /root/autodl-tmp/glaucoma_first \
  --global-checkpoint /path/to/original_global/best.pt \
  --original-checkpoint /path/to/original_local/best.pt \
  --ema-checkpoint /path/to/ema_run/best_ema.pt \
  --reference-original /path/to/paired-valid/predictions.csv \
  --reference-ema /path/to/ema_project/reports/R16-local-ema/official_valid_original_b1_ema.csv
```

必须选择新输出目录；外部文件的固定SHA、参考列、官方顺序都会验证，不复制模型或逐行标签。原版参考必须是2026-10-30同机配对完整输出（original_probability列），不是旧历史归档或翻转概率。EMA参考必须是2026-10-31最终原图B1输出，不能拿batch32缓存代替。具体摘要固定在core.py，拒绝任意参考。

运行环境沿用/root/r16-roi-env：Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128；推理阶段还要求与参考相同的NumPy2.5.4、pandas3.0.6、Pillow12.3.0。模型构建禁用预训练下载。build在Linux服务器CPU上加载/平均；check和evaluate必须CUDA。每次stage有OS锁，不并行运行，不自动重试或覆盖失败目录。

## 服务器分阶段运行

```bash
cd /root/autodl-tmp/r16_local_weight_average
mkdir -p logs
set -o pipefail
/root/r16-roi-env/bin/python -u run.py build 2>&1 | tee logs/build.log
/root/r16-roi-env/bin/python -u run.py check 2>&1 | tee logs/check.log
# 前两步均成功后再运行：
/root/r16-roi-env/bin/python -u run.py evaluate 2>&1 | tee logs/evaluate.log
```

build：核验源配置、参数/状态，固定FP32平均，保存后安全读回并严格加载；检查新局部＋冻结全图总权重≤500MB。产物在build/model.pt和build/build.json，complete.json最后写入。模型没有epoch/optimizer字段，带固定策略与两个父权重身份；不能替换原版best.pt或直接放进固定原版便携包。

check：固定种子抽16阴性＋16阳性，再加入3个已知CAM敏感样本（重复去重，最多35张）；分别跑原版、EMA和新候选。两个父模型必须与既有同机参考最大概率差≤0.001且0.5标签无变化，超差直接失败，不自动放宽或调整ROI。新候选须严格加载且概率有限。样本检查指标不与完整valid分数混比，不用于搜索系数。

evaluate：先校验build/check完成文件和源码/父权重/数据/参考哈希，再仅对新候选跑全部5849张原图实时B1。父模型指标由已验证来源的完整参考CSV复算，避免重复GPU遍历。输出evaluation/candidate.csv、report.json、status.json及最后的complete.json；报告有同集参考AUROC/AP、混淆计数、逐图耗时及硬件。

--max-minutes默认20，控制局部模型加载/预热及推理循环（初始文件验证和全图加载不计入）；超时保留部分输出并失败，不自动继续。失败目录不能重跑覆盖，先排查并准备新工程。没有训练，也不存在训练恢复。

所有时延均是该次硬件上的诊断，T4尚未验收，official_acceptance始终false。保持原图B1，禁止看到valid结果后自动尝试其他系数或翻转。若没有值得保留的收益，本项结束；有价值时再另行显式准备候选test导出和部署适配。

## 2026-10-31实际结果

源码提交39ffe837548528250f0654eaf6a9ecd7e6aa6f0d，4090D后台流程exit0，总耗时356.82秒（约6分钟）。182个FP32参数张量、0个buffer通过兼容性检查，生成新局部模型111342223字节，与冻结全图合计142589584字节。35图检查中，两父模型与参考的概率最大差都为0、硬标签变化都为0；新候选严格加载并通过有限性检查。

|完整原图valid方案|AUROC|AP|FN|FP|
|---|---:|---:|---:|---:|
|原版参考|0.9893756512|0.8644753091|13|186|
|新EMA参考|0.9893025490|0.8559727460|9|259|
|本次固定参数平均|0.9889439778|0.8578917955|9|248|

相同5849行、191阳性；父参考为原先已归档的完整输出，35图同机复现通过，并非本次重复跑完父模型。候选AUROC低于两父模型，AP低于原版，因此不采用、不导出test，不继续搜索平均系数或叠加翻转。此前等权概率平均的0.989756并非本次权重平均的预期结果，不能混用。

候选4090D端到端诊断平均53.46ms、p95 72.23ms，21/5849图超过100ms，不是T4达标证据。新模型SHA256为2009f4e106a26bdfa0429383ed59fafbc3e92459fe1fbf697af961e52d821e02。

结果已下载到工作区outputs/training_results/20261031_r16_local_weight_average/：新候选权重、build/check/evaluation完成标记、逐图输出、日志和源码均保留；40文件哈希、官方顺序及三组指标独立CPU复算通过。父模型/完整参考仍由此前私有归档保留，本包不重复打包。该输出目录含私有模型和逐图结果，不公开上传。未训练、未提交平台、未自动关机；GPU已空闲。
