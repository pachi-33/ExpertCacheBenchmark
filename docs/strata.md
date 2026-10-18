# Strata 策略适配说明

源码基准固定为 `Niko1221/Strata@296b59d2f73c8f04a805df29c2fa97655adace78`：

- [generate.cpp 的同步 adapt()](https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/src/program/generate.cpp#L8833)：候选阈值、层内配对、全局排序、交换上限和衰减。
- [expert_source.cpp 的 usage 更新](https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/src/core/expert_source.cpp#L3716)：对 n_tok × top_k 的每个出现项加一，包含 batch 内重复专家。

实现规则：

1. 初始化按外部 profile 的全局 `(layer, expert)` 热度降序预装，直到预算或层配额不足；在线 usage 从零开始。
2. 每个路由出现项增加 1，使用 float32 usage。
3. 默认每 4 个 decode step 调整。各层分别取 usage ≥ 2 的非驻留候选，按热度降序排序；驻留且未 pin 的专家按热度升序排序。
4. 将两组按位置配对；候选 usage ≥ 被替换者 usage + 1.5 才交换。这是加法差值，包含等号。
5. 将各层候选交换合并，按热度增益排序，最多执行 96 次。始终同层交换，保持各层驻留数量。
6. 每轮调整之后，全部 usage 乘 0.7；没有符合条件的交换也会衰减。adapt_every=0 或 max_swaps=0 禁用调整。
7. 同分情况按 layer/expert ID 明确排序；C++ 原实现未给同分项指定稳定顺序。

Strata 需要 profile、preload 或 restore 提供初始驻留。空池没有可交换槽位，因此正容量下的 cold 初始化会报错，避免把全 miss 冒充已实现适应式缓存。容量为零可用来测试全 miss。只在层内专家大小相同时允许一换一。

## 回放与真实 Strata 的差别

当前 trace 只含已输出 token 的路由，没有原始 MTP 验证窗口和拒绝草稿；本实现明确以 decode step 代替 verify window 作为调整周期。改变 batch size 会改变每轮收集到的路由数量。

这里复现的是同步缓存驻留决策。当前 Engine 假设主存保留全部权重副本，每个 miss 通过临时空间传入计算设备，适应式换入也单独计费。真实 Strata 还可能在 CPU 上执行 miss、执行 D2H 交换、跨 GPU 分配或者异步搬运。因而本框架报告的 transfer units 是统一搬运模型下的策略成本，不能直接解释为原生 Strata 的 PCIe 字节或性能。

## 可重复的对照

```bash
expertcache profile dataset/router_results_sample100/benchmark.json \
  --max-requests 20 --output runs/profile-train20.json
expertcache run configs/strata-smoke.json --output runs/strata-smoke
expertcache run configs/lfu-profile-smoke.json --output runs/lfu-profile-smoke
```

前 20 个请求生成 profile，接下来的 4 个请求各回放 32 token。两个策略共用预算、profile、请求和调度。完整的 `strata-eval.json` / `lfu-eval.json` 评估其余 80 个请求的全部 512 token。数据源按原始索引排序，首 20 个请求属于 math；这是明确的固定划分，不能把此 smoke 当作无偏的算法排名。
