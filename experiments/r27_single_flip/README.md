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
