# Belady / MIN 离线基线

内置 `policy.name="belady"`，必须显式设置 `visibility="oracle"`。使用全局缓存池、等大小专家、相同的 profile 初始驻留，未来路由通过 `ctx.oracle.batches()` 获取；普通 causal 策略无法使用该入口。

每个 batch-layer 先以共同快照标注命中，再更新所有当前专家的下一次使用位置。从“原驻留 ∪ 当前需求”中保留下一次使用最近的专家；新需求比所有驻留专家更晚使用时，经 workspace 临时服务并绕过常驻池。当前 batch 的多个请求使用同一专家只对应一次需求加载，当前层的加载不会反过来增加当前命中。

缓存可跨层分配，不受 Strata 初始化后的层内槽位约束。没有预取、固定 pin 或大小不同的专家；未再次使用的专家视为下一次使用在无穷远。同下一次使用时间时优先保留现有驻留，淘汰并列专家时按专家键确定顺序。

## 上限的准确含义

该实现最小化固定初始驻留、相同 batch 快照、等大小全局池、按需准入且允许 bypass 条件下的**去重专家加载次数**。对应复用区间的单位收益选择：容量不足时放弃结束最晚的区间。小规模测试用枚举所有合法缓存状态的动态规划验证这一目标的最优性；完整实验再独立检查每个 batch-layer 的快照、搬运计费和保留集合的下一次使用边界。

报表中的逐 request/token/layer 命中率仍按原有口径计算，但它不是严格的加权命中率上界：同一缺失专家若被 4 个请求使用，加载代价是 1，命中收益却是 4。不同的搬运最优计划可能得到不同的加权命中率。允许提前加载的算法也可通过额外搬运提高命中率，因此应同时比较 evaluation 的总搬运量。

经典 MIN 的最优性依赖缓存模型和目标函数；关于 Belady、适用条件及更一般的离线上界，可参考原研究 [Practical Bounds on Optimal Caching with Variable Object Sizes](https://arxiv.org/abs/1711.03709)。这里没有把多 batch 的命中统计直接转换为逐请求的经典分页定理。

## 运行

```bash
# 使用与 LFU / Strata 相同的独立 profile
expertcache profile dataset/router_results_sample100/benchmark.json --max-requests 20 --output runs/profile-train20.json
expertcache run configs/belady-eval.json --output runs/belady-eval

# 给原 12 组实验追加 Belady 六档容量，保留原始实验指纹
PYTHONPATH=src python scripts/capacity_sweep.py --policies belady --extend-report docs/results/capacity-sweep-20261018.json --output runs/belady-sweep --workers 3
PYTHONPATH=src python scripts/verify_capacity_sweep.py runs/belady-sweep/results.json --output runs/belady-sweep/verification.json
```

已有 profile 时直接复用。`--extend-report` 要求相同数据和 profile 指纹，并拒绝覆盖已有策略；原有结果保持原样。新增策略代码可以有不同 Git revision，核验要求同一策略的六档实验使用同一 revision 和源码 SHA-256。完整运行和封存 detail 均由普通 Benchmark 生命周期生成。

oracle 提供当前阶段完整的实际调度序列，包括打乱、请求偏移、token 限制和 continuous batching；oracle 回放复用同一份调度计划，避免二次调用有状态调度器改变顺序。warmup 与 evaluation 分开建索引，restore 根据新阶段的路由重新建立未来索引。
