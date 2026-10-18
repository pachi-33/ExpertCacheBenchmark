# Batch size 对比：BS=1 与 BS=4

沿用前 20 请求 profile，评估其余 80 请求 × 512 token × 40 有效层；top-k=6。每专家 1 unit，workspace=1 unit。
每组 1,638,400 条逐层标注；新增 BS=1 共 12 组、19,660,800 条，全部完成独立核验。
逐对核验配置（仅 batch size 不同）、请求顺序、完整初始缓存与策略状态、数据/profile 指纹和策略源码；历史结果保持原样。

## 平均命中率

| 缓存 | 策略 | BS=1 | BS=4 | 差值（百分点） |
|---|---|---:|---:|---:|
| 10% | strata | 57.48% | 48.77% | +8.71 |
| 10% | belady | 78.16% | 68.28% | +9.88 |
| 15% | strata | 66.20% | 58.39% | +7.81 |
| 15% | belady | 84.60% | 77.24% | +7.36 |
| 25% | strata | 76.69% | 71.76% | +4.93 |
| 25% | belady | 91.41% | 87.00% | +4.41 |
| 30% | strata | 80.21% | 76.72% | +3.50 |
| 30% | belady | 93.45% | 90.03% | +3.42 |
| 35% | strata | 83.22% | 80.91% | +2.31 |
| 35% | belady | 94.97% | 92.33% | +2.63 |
| 50% | strata | 90.32% | 90.14% | +0.18 |
| 50% | belady | 97.70% | 96.67% | +1.03 |

## Evaluation 权重搬运量

包含 demand 与 adaptation 等全部 evaluation 加载，初始化单列。相同输出 token 数下直接比较总量；差值为候选相对基线。

| 缓存 | 策略 | BS=1 units | BS=4 units | 变化 | BS=1 units/token | BS=4 units/token |
|---|---|---:|---:|---:|---:|---:|
| 10% | strata | 4,768,736 | 5,062,875 | -5.81% | 116.42 | 123.61 |
| 10% | belady | 2,147,444 | 3,014,275 | -28.76% | 52.43 | 73.59 |
| 15% | strata | 3,787,541 | 4,192,990 | -9.67% | 92.47 | 102.37 |
| 15% | belady | 1,514,266 | 2,177,182 | -30.45% | 36.97 | 53.15 |
| 25% | strata | 2,588,616 | 2,941,788 | -12.01% | 63.20 | 71.82 |
| 25% | belady | 844,149 | 1,251,830 | -32.57% | 20.61 | 30.56 |
| 30% | strata | 2,191,376 | 2,467,924 | -11.21% | 53.50 | 60.25 |
| 30% | belady | 643,686 | 962,182 | -33.10% | 15.71 | 23.49 |
| 35% | strata | 1,854,236 | 2,060,507 | -10.01% | 45.27 | 50.31 |
| 35% | belady | 494,835 | 741,082 | -33.23% | 12.08 | 18.09 |
| 50% | strata | 1,063,089 | 1,099,579 | -3.32% | 25.95 | 26.85 |
| 50% | belady | 226,478 | 322,721 | -29.82% | 5.53 | 7.88 |

## 去重与适应性换入

| 缓存 | 策略 | BS=1 demand | BS=4 demand | BS=1 adaptation | BS=4 adaptation | BS=4 合并 miss 数 |
|---|---|---:|---:|---:|---:|---:|
| 10% | strata | 4,180,073 | 4,841,899 | 588,663 | 220,976 | 194,588 |
| 10% | belady | 2,147,444 | 3,014,275 | 0 | 0 | 104,220 |
| 15% | strata | 3,323,087 | 3,953,548 | 464,454 | 239,442 | 136,911 |
| 15% | belady | 1,514,266 | 2,177,182 | 0 | 0 | 60,257 |
| 25% | strata | 2,291,213 | 2,700,771 | 297,403 | 241,017 | 75,420 |
| 25% | belady | 844,149 | 1,251,830 | 0 | 0 | 25,851 |
| 30% | strata | 1,945,179 | 2,231,634 | 246,197 | 236,290 | 57,135 |
| 30% | belady | 643,686 | 962,182 | 0 | 0 | 17,760 |
| 35% | strata | 1,649,673 | 1,833,383 | 204,563 | 227,124 | 43,446 |
| 35% | belady | 494,835 | 741,082 | 0 | 0 | 12,471 |
| 50% | strata | 951,716 | 952,111 | 111,373 | 147,468 | 17,421 |
| 50% | belady | 226,478 | 322,721 | 0 | 0 | 4,581 |

## 结果解读

- 六档容量下，BS=1 的命中率均更高。Strata 的优势随容量增大，从 10% 时的 **+8.71 个百分点** 缩小到 50% 时的 **+0.18 个百分点**。
- 相同 40,960 个输出 token 下，Strata 总搬运减少 **3.32%–12.01%**，Belady 减少 **28.76%–33.23%**。这里同时改变了请求交错顺序、batch 内去重和按 step 触发的策略时钟，不能将差值单独归因于其中某一项。
- Strata 在 10% 缓存下，BS=1 的 demand 从 4,841,899 降至 4,180,073 units，但 adaptation 从 220,976 升至 588,663 units，抵消了一部分收益，因此总搬运只下降 5.81%。
- Strata 在 50% 缓存下，两种 BS 的 demand 几乎相同（951,716 与 952,111 units）；总搬运的差异主要来自 adaptation，BS=1 为 111,373，BS=4 为 147,468 units。
- Strata 保持每 4 step 调整一轮，因此 BS=1 共 10,240 轮，BS=4 共 2,560 轮；这是保留原参数的 batch 对比。

## 比较口径

- BS=1 顺序完成每个请求；BS>1 交错推进多个请求。缓存跨请求保留，因此 batch size 同时改变全局访问顺序与局部性。
- Strata 参数保持不变：每 4 个 decode step 调整一次，decay=0.7、min_count=2、min_gain=1.5、max_swaps=96。没有按总 token 数重新缩放策略时钟；具体调整次数保留在 JSON 的 policy_metadata 中。
- BS>1 同层重复 miss 可合并为一次 demand 加载；命中率仍按每个请求的 top-k 分别计数。不能只凭命中率推算权重搬运量。
- Belady 针对各自调度后的访问序列，优化等大小全局池的去重需求加载，允许 bypass、固定初始驻留。BS>1 的 occurrence 加权命中率不保证最优。
- 结果是离线 cache 命中与权重搬运统计，不是实际模型吞吐、TPOT 或 MTP 加速。

## 耗时与核验

本轮 BS=1 扫描墙钟耗时 **676.67 秒**，独立核验部分与扫描并行。各组进程耗时在配套 JSON 中保留，包含数据校验、初始化、回放和 summary；并行运行下不能视为独占机器的算法性能排名。
核验覆盖全部路由、记录顺序、覆盖范围、hit_mask/hit_count、均值与降序分位数、初始化和报告校验和；Belady 额外从缓存事件重建每个快照并检查每一步下一次使用的保留/排除边界。

配套 JSON 保留每组完整命中率分位数、需求/适应性搬运、batch 合并量、来源指纹和配对检查结果。

## 复现

需要 `runs/profile-train20.json`。若已保留历史 BS=4 的完整运行目录，可复用其报告，省略 BS=4 重跑；只有精简报告时需先重跑对应配置。不同 batch size 写入不同目录。

```bash
PYTHONPATH=src python scripts/capacity_sweep.py --batch-size 4 --policies strata belady --workers 3 --output runs/compare-bs4
PYTHONPATH=src python scripts/capacity_sweep.py --batch-size 1 --policies strata belady --workers 3 --output runs/compare-bs1
PYTHONPATH=src python scripts/verify_capacity_sweep.py runs/compare-bs4/results.json --output runs/compare-bs4/verification.json
PYTHONPATH=src python scripts/verify_capacity_sweep.py runs/compare-bs1/results.json --output runs/compare-bs1/verification.json
PYTHONPATH=src python scripts/compare_batch_sizes.py --baseline runs/compare-bs4/results.json --candidate runs/compare-bs1/results.json --verification runs/compare-bs1/verification.json --output runs/bs1-vs-bs4
```

完整原始输出及输入副本已归档到项目的 `runs/archive/20261018/`：

| 目录 | 内容 |
|---|---|
| `bs1-sweep/` | 本次 Strata / Belady BS=1 的 12 组完整输出 |
| `bs4-lfu-strata/` | 历史 LFU / Strata BS=4 的 12 组完整输出 |
| `bs4-belady/` | 历史 Belady BS=4 的 6 组完整输出 |
| `inputs/` | 相同路由数据集与 `profile-train20.json` 的完整副本 |

共 30 组、49,152,000 条标注，3,359 个文件、5,389,460,730 bytes。复制后逐文件核对 SHA-256，并对全部运行的 manifest、summary 和已封存 detail 的索引校验和复核。清单见 [完整归档索引](benchmark-archive-20261018.json)，记录原始路径、归档路径、文件大小与 SHA-256。

Git 保存统计报告、核验报告、归档清单和复现脚本；大体积原始输出保存在被 Git 忽略的 `runs/`。归档按原字节保留 manifest/config 中的历史路径。需要重新运行时，可将 `--dataset` 指向 `runs/archive/20261018/inputs/router_results_sample100/benchmark.json`，`--profile` 指向 `runs/archive/20261018/inputs/profile-train20.json`。

- [BS=1 原始汇总](capacity-sweep-bs1-20261018.json)
- [BS=1 独立核验](capacity-sweep-bs1-20261018-verification.json)
- [BS=1 / BS=4 配对指标与检查](batch-size-bs1-vs-bs4-20261018.json)
- [历史 BS=4 结果及来源说明](capacity-sweep-20261018.md)

本次未修改策略或回放核心；沿用历史 BS=4 的版本与来源记录，不重写其源码指纹。
