# 缓存容量扫描：完整留出集

比例按 40 个有效层 × 256 专家 = 10,240 个专家计算；每专家 1 unit，另有 1 unit workspace。
相同的前 20 请求 profile 初始化，评估其余 80 请求 × 512 token，batch=4。每组 1,638,400 次判断，12 组共 19,660,800 次。
所有判断逐项核对原始路由与覆盖范围，均值及降序分位数从 hit_count 直方图独立重算。

| 缓存比例 | 专家数 | LFU 平均命中率 | Strata 平均命中率 |
|---|---:|---:|---:|
| 10% | 1024 | 20.85% | 48.77% |
| 15% | 1536 | 27.50% | 58.39% |
| 25% | 2560 | 39.44% | 71.76% |
| 30% | 3072 | 45.32% | 76.72% |
| 35% | 3584 | 50.82% | 80.91% |
| 50% | 5120 | 65.99% | 90.14% |

## 降序分位数

P10/P50/P90/P99 按命中率从高到低排序，使用 linear 插值；P50 与中位数一致。

| 缓存比例 | 策略 | 平均 | P10 | P50 | P90 | P99 | 中位数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 10% | lfu | 20.85% | 50.00% | 16.67% | 0.00% | 0.00% | 16.67% |
| 10% | strata | 48.77% | 83.33% | 50.00% | 16.67% | 0.00% | 50.00% |
| 15% | lfu | 27.50% | 50.00% | 16.67% | 0.00% | 0.00% | 16.67% |
| 15% | strata | 58.39% | 83.33% | 66.67% | 33.33% | 0.00% | 66.67% |
| 25% | lfu | 39.44% | 66.67% | 33.33% | 16.67% | 0.00% | 33.33% |
| 25% | strata | 71.76% | 100.00% | 66.67% | 50.00% | 16.67% | 66.67% |
| 30% | lfu | 45.32% | 83.33% | 50.00% | 16.67% | 0.00% | 50.00% |
| 30% | strata | 76.72% | 100.00% | 83.33% | 50.00% | 16.67% | 83.33% |
| 35% | lfu | 50.82% | 83.33% | 50.00% | 16.67% | 0.00% | 50.00% |
| 35% | strata | 80.91% | 100.00% | 83.33% | 50.00% | 33.33% | 83.33% |
| 50% | lfu | 65.99% | 100.00% | 66.67% | 33.33% | 16.67% | 66.67% |
| 50% | strata | 90.14% | 100.00% | 100.00% | 66.67% | 50.00% | 100.00% |

## 复现与原始输出

输入数据和 profile 的本地副本均已校验指纹一致。完整输出位于 `/private/tmp/ExpertCacheBenchmark-capacity-sweep`，各组包含 manifest、状态、detail 与 summary。
`capacity-sweep-20261018.json` 保存全部 summary 与运行指纹；`capacity-sweep-20261018-verification.json` 保存 12 组独立核验结果。
脚本：`scripts/capacity_sweep.py`；校验：`scripts/verify_capacity_sweep.py`。
本机 Desktop 的 FileProvider 将文件标记为 dataless，导致首次输出及部分源码指纹读取异常。有效 detail 已改存在本地临时目录；最后两组使用相同代码的本地副本完成标准后处理，全部 manifest 状态为 complete。
报告保留原始源码指纹差异，并核对所有实验的 Git revision 与各策略源码 SHA-256 一致；没有重写历史指纹。

```bash
PYTHONPATH=src python scripts/capacity_sweep.py --output /private/tmp/ExpertCacheBenchmark-capacity-sweep
PYTHONPATH=src python scripts/verify_capacity_sweep.py /private/tmp/ExpertCacheBenchmark-capacity-sweep/results.json --output /private/tmp/ExpertCacheBenchmark-capacity-sweep/verification.json
```
