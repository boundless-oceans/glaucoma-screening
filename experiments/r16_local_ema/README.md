# R16局部分支EMA对照

本实验只增加局部参数的指数移动平均（EMA），全图E32 epoch7和CAM/ROI规则冻结。推理时分别加载普通或EMA局部权重，不同时运行二者，因此EMA本身不增加推理模型数量。服务器工程检查、正式资产、15轮训练和完整B1验证已完成；EMA未超过冻结原版，保留原版作为基线。

## 对照范围

复用相邻r16_roi_consistency的已验收配方：train/valid CAM资产均为B1、坐标17位写入和round_trip读回，局部ConvNeXt-Tiny/384从指定通用ImageNet22K→1K初始化；保持原增强、自然采样、约30的BCE正类权重、分组学习率、warmup/余弦和原图50/50 raw-logit融合。仅官方train参与梯度，valid选模，test不读取。

原版队长训练资产与此配方不同，之前ROI一致性变体也未超过原版，不能把本实验与队长分数的差异全部归因于EMA。**主要对照是同次训练、同轮次的普通参数与EMA参数。** 同时保留双方独立最佳作为选模诊断；最佳轮次可能不同，须分开解释。既有valid已反复研究，不是独立盲测。

EMA衰减固定0.999，不搜索。初始化在工程probe复位以后、正式第一步之前，复制所有状态且不消耗随机数。仅在成功optimizer.step后更新，AMP溢出跳步不更新；参数FP32，所有buffer直接复制（包括浮点统计和整数计数）。影子模型保持eval、无梯度、不进入优化器。普通分支仍控制patience=5；EMA不改变停止时机。原local_epochs=20及学习率日程不变，预算从5400增至10800秒以容纳额外验证，实际时长尚待小样本测量。

## 本地轻量检查

在仓库根目录执行：

```bash
python experiments/r16_local_ema/tests/test_ema.py
```

6项CPU检查已通过：参数递推、buffer复制、结构/策略拒绝、源码锚点、完整工程准备/哈希和失败不发布目录。不导入PyTorch、不加载模型。原vendor和旧实验源码不修改；新脚本在临时目录构建并校验，成功后才发布新工程，拒绝覆盖已有目录。

2026-10-31：提交1b0b3b460d58203379774f71e76b4567d3ce8400的源码包在4090D完成check，约30.48秒、exit_code=0。32/32图裁剪像素相同，随机局部模型完成2次成功更新且EMA计数一致；微型张量检查确认1次强制AMP跳步不更新EMA、buffer复制、普通权重不被EMA修改和状态读回；两轮合成训练的双验证、独立选模、两种best及last内EMA状态读回均通过。合成分数不代表比赛效果，随机局部权重已丢弃，正式runs为空，未生成training_started.json。

工程检查报告已下载到工作区outputs/training_results/20261031_r16_local_ema_check/，29个文件哈希通过复核，生成控制器/runner/helper与本次提交相符。检查阶段通用初始化仅完成文件哈希核验；后续已在正式train中完成加载。服务器独立工程为/root/autodl-tmp/r16_local_ema_20261031，正式流程结果见文末。

## 服务器准备

使用已验收的独立/root/r16-roi-env（Python3.12、torch2.10.0+cu128、torchvision0.25.0+cu128及原训练依赖，包含pandas、Pillow、NumPy、scikit-learn、PyYAML），不要套用旧B2训练的glaucoma-cloud环境。上传源码时保留相邻r16_roi_consistency目录。本地也可执行prepare，它仅进行文件操作和哈希验证，不反序列化模型。

```bash
python experiments/r16_local_ema/scripts/prepare.py \
  --output /root/autodl-tmp/r16_local_ema \
  --data-root /root/autodl-tmp/glaucoma_first \
  --global-checkpoint /path/to/original_E32_epoch7/best.pt \
  --local-initializer /path/to/convnext_tiny_22k_1k_384.pth
```

数据根目录为含train.csv、valid.csv和图片目录的官方解压目录。全图权重SHA256必须为b130ffe685e8e945b18184abbcc159135e87a537d2a5824222d927791cd5b6d2；通用初始化必须为f22f8850dcded245f4ae3bc402bbde9c6f5e4e23e76dfaea3aadb75aa9d3705a。无需E33比赛最佳权重，它只是既有基线，不作为本次初始化。不使用医学类预训练或额外数据。

## 分阶段运行，不能直接启动正式训练

```bash
cd /root/autodl-tmp/r16_local_ema
mkdir -p logs
set -o pipefail
/root/r16-roi-env/bin/python -u run.py check 2>&1 | tee logs/check.log
```

check需要CUDA服务器：沿用32图CAM/像素检查和随机局部真实更新，增加EMA更新计数；另运行微型真实张量检查（FP32、无共享存储、buffer、强制AMP跳步、EMA检查点读回），以及生成控制器的两轮合成训练/双验证/选模/保存检查。合成数据不会混入正式清单，临时权重检查后清除。这些检查不证明通用初始化已验收或分类效果良好。检查通过才生成正式资产：

```bash
/root/r16-roi-env/bin/python -u run.py assets 2>&1 | tee logs/assets.log
```

与旧工程来源指纹不同，不能直接复制旧完成标记绕过新检查；资产继续由冻结的全图模型生成。检查与资产完成后，先按实测耗时确认预算，再显式启动训练：

```bash
nohup /root/r16-roi-env/bin/python -u run.py train \
  > logs/train.log 2>&1 &
tail -f logs/train.log
```

每个完整epoch增加一次EMA valid前向，训练会比普通分支更久，不提前承诺时长。输出位于新工程reports/R16-local-ema/和runs/，不覆盖旧实验。普通patience或轮数上限结束返回0；预算/磁盘提前停止且有完整普通验证时返回3，local_result.json标status=stopped_incomplete及停止原因。没有完整普通验证则报错退出。**沿用one-shot：不支持中间batch或预算中断续训，不能重复同一train命令当恢复；已有training_started.json会拒绝重启。** 保存EMA状态用于审计和后续开发，不代表完整训练恢复功能。不要用旧train.sh的--max-hours或--resume。

## 文件与最终复核

- best.pt、metrics.json、valid_predictions.csv：普通局部权重按固定融合AUROC、AP选优，原语义不变。
- best_ema.pt、ema_metrics.json、valid_ema_predictions.csv：EMA局部权重独立按同样指标选优。
- paired_history.json：只记录普通与EMA均完成的同轮完整验证，含epoch、成功更新数、双方指标和累计时间，是主要对照。
- valid_raw_XX/valid_fused_XX及valid_ema_raw_XX/valid_ema_fused_XX.csv：普通/EMA局部及固定融合逐图输出。
- last.pt/partial.pt：普通权重、优化器、scheduler、scaler和EMA状态/计数。不是自动可恢复入口。
- completion.json、reports/R16-local-ema/local_result.json：停止原因、普通完整轮数、配对完整轮数、两份best的哈希；提前停止时须标明实验未完成，不与完整实验混比。

训练结束后显式运行原图B1复核（各自最佳可能不是同一epoch）：

```bash
/root/r16-roi-env/bin/python -u run.py evaluate 2>&1 | tee logs/evaluate.log
```

分别读取best.pt/best_ema.pt，检查权重哈希，使用相同冻结全图进行实时CAM/ROI推理；分别写official_valid_original_b1_raw/ema.csv与对应metrics，最终results.json包含两者。不会临时替换best.pt别名，不读取test或导出提交；完整results.json存在时拒绝重复覆盖。评估失败则不写最终results.json，先排查后处理。

当前r16_inference便携入口严格绑定原版两份权重，不会自动接受本实验新权重；不要覆盖原版模型文件。只有本实验显示值得采用，才另行适配候选导出和部署。

T4测速是独立待办，4090D只用于工程验收和训练对照。EMA在T4上同样需要实测，不预先宣称满足100ms。若无明确收益，保留队长原版及既有候选，不用本次微小valid变化保证平台涨分。

## 2026-10-31正式实验结果

源码提交1b0b3b4。完整资产51396/5849图，train/evaluate均exit0；15轮普通/EMA配对验证、24095次成功更新后，普通分支连续5轮未超过第10轮而正常patience停止。资产约11.28分钟、训练76.52分钟、两份最佳的B1验证11.03分钟，整个流程约98.83分钟，未触发预算中断。

|方案|独立选优轮次|最终B1 AUROC|最终B1 AP|FN|FP|
|---|---:|---:|---:|---:|---:|
|同次普通权重|10|0.9890619593|0.8517480648|9|239|
|同次EMA权重|11|0.9893025490|0.8559727460|9|259|
|原版冻结参考|原E33第15轮|0.9893756512|0.8644753091|13|186|

均为相同官方valid 5849张、191阳性、原图及固定全图/局部logit融合；原版参考来自此前同机归档，不是同次训练的因果对照。前两行是各自独立选优，不能称为同一epoch比较；同轮轨迹见paired_history.json。缓存batch32选模指标与实时B1复核有微小数值差异（EMA选模AUROC约0.989305325，最终0.989302549），统一采用最终B1作为交付汇报值。

EMA略优于本轮普通版，但AUROC、AP仍未超过原版，不替换基线，也不生成test提交。0.5阈值下本轮EMA较普通版多20个FP，不能把AUROC微涨解释为所有指标都更好。两模型4090D平均诊断耗时约54.01/56.56ms，不是T4验收。

权重（普通best、EMA best、含EMA状态的last）、日志、来源、资产、15轮逐图验证及最终B1输出已下载到工作区outputs/training_results/20261031_r16_local_ema/。该归档含私有清单与权重，不公开上传；原始冻结全图权重仍由既有私有备份保存，不重复打包。没有自动关机或提交平台。
