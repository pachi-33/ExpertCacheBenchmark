# 完整 4K Router 数据集：Strata / BS=1 / Cache=35%

全部 4,000 条请求先用 seed=20261018 打乱，前 20 条构建 profile，剩余 3,980 条用于评估。缓存由 profile 初始化，跨请求、跨任务持续保留；任务统计由 summary hook 在同一次回放结束后聚合。

- 数据记录的模型标识：`DeepSeek-V4-Flash-w8a8-mtp`，40 个有效 MoE 层（3–42），每层 256 专家，top-k=6。每请求生成并记录 512 个输出 token，不包含 prompt；全部 4,000 条请求的 `completion_tokens=512`，trace 形状为 `[512, 43, 6]`。
- 常驻池：3,584 / 10,240 virtual expert units（35%）；workspace 另为 1 unit。
- Strata：每 4 step 适应一次，decay=0.7、min_count=2、min_gain=1.5、max_swaps=96；每层初始槽位数由全局 profile 选取结果决定，适应时在层内替换。
- 统计单位：每个 request/token/layer 的 hit_count / 6。P10、P50、P90、P99 从高到低排名，分别对应升序 q90、q50、q10、q1，使用线性插值。平均值按全部专家访问计数，与逐层均值相等。
- 评估规模：2,037,760 token，81,510,400 条逐层标注，489,062,400 次专家访问。

## 命中情况

| 任务 | 评估请求 | P10 | P50 | P90 | P99 | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| 总体 | 3,980 | 100.00% | 83.33% | 66.67% | 33.33% | 82.91% |
| agent | 994 | 100.00% | 83.33% | 50.00% | 33.33% | 82.99% |
| chat | 993 | 100.00% | 83.33% | 66.67% | 33.33% | 84.38% |
| coding_swe | 996 | 100.00% | 83.33% | 50.00% | 33.33% | 80.43% |
| math | 997 | 100.00% | 83.33% | 66.67% | 33.33% | 83.83% |

Profile 请求分布：agent=6、chat=7、coding_swe=4、math=3。这 20 条不计入评估。

任务组共享整个回放期间的缓存状态；这些分组数据不等价于将各任务独立重跑。与旧 sample100 实验的 profile、请求集合及顺序均不同。

## 权重搬运量与耗时

- Profile 初始化：3,584 units。
- 评估 demand：83,600,604 units；adaptation：11,010,083 units。
- 评估合计：94,610,687 units，平均 46.4288 units/token。
- 完整 benchmark 墙钟耗时 **399.915 秒**，包括数据校验、profile、编译、回放、detail 封存、summary 和完整缓存事件扫描；其中准备耗时 7.643 秒。
- 独立核验与归档在 benchmark 之后执行，其耗时不包含在上述数字中。
- 耗时是本机离线框架执行时间；权重使用虚拟单位，没有模拟实际硬件时序。

## 执行与核验

为完成 4K 规模回放，使用可选的 C++ Strata BS=1 后端，保留标准 detail、缓存事件、profile 与最终状态。通用 Python runner 保留为参考实现。

- 运行前：42 项单元测试通过；另取真实数据的 2 个完整请求，与 Python 逐条对照 40,960 条标注、105,939 条缓存事件，以及初始/最终状态和 summary，全部一致。
- 运行后：核对全部 81,510,400 条标注的原始路由、覆盖顺序、hit_mask 与 hit_count、总体及任务直方图、降序分位数、profile/eval 分离和初始槽位。
- 完整事件扫描核对 200,238,625 条缓存事件的序号和传输配对；demand 与 detail miss 数完全一致。
- 不声称整组 3,980 请求与 Python 双跑对照；全量核验和两请求参考对照分别保留证据。
- 报告记录源码指纹、C++ 源码及二进制 SHA-256、编译器版本、数据集与 profile 指纹。`framework_revision` 为运行时的 Git 基线，实际未提交源码以归档的 `source-snapshot/` 为准。

## 复现与原始记录

```bash
PYTHONPATH=src python scripts/run_strata_bs1.py \
  --dataset dataset/cascade_balanced_4k/router_results_uint8/benchmark.json \
  --output runs/strata-full4k-bs1-cache35-reproduce \
  --cache-percent 35 --seed 20261018 --profile-requests 20 --backend native
```

需要 C++17 编译器及 zlib。使用 `--backend python` 可运行参考实现，耗时会显著增加。

完整输出保存在本地 `runs/strata-full4k-bs1-cache35-shuffled-20261018/`，共 5,045 个文件、7,224,034,828 bytes。所有文件复制后逐个核对 SHA-256。包括标准 detail 和缓存事件、summary、profile、请求顺序、配置、manifest、状态快照、执行源码及参考对照记录。`runs/` 由 Git 忽略，精简结果和校验清单纳入版本管理。

原始 manifest/config 按字节保留 `/private/tmp/` 路径，归档清单提供到仓库 `runs/` 目录的映射；复现时使用上面的新输出目录。

- [结构化结果](strata-full4k-bs1-cache35-shuffled-20261018.json)
- [全量核验](strata-full4k-bs1-cache35-shuffled-20261018-verification.json)
- [逐文件归档清单](strata-full4k-bs1-cache35-shuffled-20261018-archive.json)
- [本地原始 summary](../../runs/strata-full4k-bs1-cache35-shuffled-20261018/strata-35pct/summary.json)
- [Python 对照证据](../../runs/strata-full4k-bs1-cache35-shuffled-20261018/validation/native-equivalence.json)
