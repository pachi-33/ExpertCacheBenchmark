# ExpertCacheBenchmark

离线路由回放框架：按模型前向顺序公开信息，通过生命周期 hooks 执行缓存策略，输出逐层标注和可独立重算的 summary。第一版只统计命中与权重搬运，不预测 TPOT。

## 快速运行

Python 3.11+；依赖 NumPy 和 Pydantic 2。

```bash
python -m pip install -e .
expertcache validate dataset/router_results_sample100/benchmark.json
expertcache run configs/lfu-smoke.json --output runs/lfu-smoke
expertcache postprocess runs/lfu-smoke
expertcache export-detail runs/lfu-smoke --output runs/lfu-smoke/access.jsonl
python -m unittest discover -s tests -v
```

也可以使用 `PYTHONPATH=src python -m expertcache ...`，无需安装项目。

`lfu-smoke.json`：4 个请求，每个 32 个 token，batch=4，1024 units 常驻池，1 unit 临时空间。`lfu-full.json` 回放全部 100 × 512 × 40 条逐层判断。输出目录必须不存在，避免覆盖已有实验。

## 实验语义

- 一个判断 = 一个 request/token/layer 的 top-k 集合。`hit_count / top_k` 是一个命中率样本。
- 同一个 batch-layer 用同一份加载前快照判定全部行。两个请求对同一个未驻留专家都记 miss，实际只加载一次。
- 本层先消费快照中的命中专家，再串行服务缺失专家；因此需求并集可以超过缓存大小。未缓存的专家用 workspace 临时加载，完成后释放。
- 默认每个专家 1 个虚拟 unit；缓存与 workspace 分别限制。支持逐专家大小、固定层配额、动态层配额、pin/unpin。
- 缓存在同一 run 中跨请求共享，run 之间独立。调度是 FIFO continuous batching，每步每个活跃请求处理一个 token；结束的槽位下一步补入。
- 按需加载不能改变本次命中标注。提前加载可以影响后续命中；所有实际加载都计入搬运量。
- v1 所有加载立即完成，事件记录逻辑顺序、transfer ID 和 `simulated_time=null`。`TransferBackend` 是时序模拟接入点；未来还需实现完成队列与时钟调度，不能把当前结果解释成时延。
- 全部权重有主存后备副本，淘汰无需写回。统计表示该内存模型下的搬运量。

## 生命周期与扩展

```text
校验数据与配置 → 创建空池与策略 → on_initialize / restore
→ preload / profile / warmup → 保存 initial_state
→ on_run_start
→ [on_request_start → on_step_start
   → [on_layer_start → 公开 routes → 固定 probe
      → on_routes_ready → 执行计划并补齐 demand
      → on_layer_end → 提交逐层标注]
   → on_step_end → on_request_end]
→ on_run_end → 保存 final_state → 封存 detail → summary hooks
```

每个决策 hook 返回 `CachePlan`，Engine 执行后再进入下一阶段。`on_actions_applied` 是只读反馈。所有 hook、动作、Context 接口见 `src/expertcache/api.py`。自定义算法继承 `CachePolicy`，配置 `policy.name="your_module:YourPolicy"`；参数从 `policy.params` 传给构造函数。

Context 提供只读缓存、专家目录、本阶段已公开的路由历史。`context.history.routes(...)` 提前读取会失败。`visibility="oracle"` 才提供完整选中请求的 Oracle 视图，并记入 manifest。Python 插件是受信任的代码；这是因果接口约束，不是隔离恶意插件的安全沙箱。请求的未来长度和原始 trace 不进入普通 Context。

初始化支持 cold、显式 preload、独立 profile、warmup、完整 restore。profile/warmup 与评估请求 ID 必须不重叠。profile 只决定预装顺序，LFU 在线计数从零开始。warmup 使用相同前向 hooks，保留策略状态；正式阶段的 `on_run_start` 在 warmup 后触发。restore 需要模型、池配置、策略配置相同，恢复缓存及算法状态；恢复已有驻留不额外虚构加载。初始化、预热、评估搬运分别统计。

Summary hook 实现 `namespace` 和 `process(detail_reader, run_manifest) -> dict`，通过 `summary_hooks` 或 `postprocess --hook module:Class` 加载。namespace 不允许冲突。后处理失败时保留已封存 detail，可修复插件后单独重跑。Scheduler 也可以通过 Python 的 `Benchmark(..., scheduler=...)` 注入，实现 `steps()` 接口。

## 数据与输出契约

Pydantic 契约在 `contracts.py`；运行 `expertcache schemas --output docs/schemas` 导出 JSON Schema。输入 manifest 的 route 文件为相对路径、无 pickle 的 `.npy`，轴固定 `[token, layer, topk]`。支持 uint8/uint16/uint32；层 ID 保留原编号。

当前数据第 0–2 层均是零占位，不能解释成有效专家选择。`benchmark.json` 明确只评估第 3–42 层，逐项注明排除原因；原始 43 层数据保持完整。记录仅含已输出 token 的路由，不含 prefill 或被拒绝的 MTP 草稿，无法重建这些前向。

```text
run/
  manifest.json                 # 完整配置、数据/源码指纹、版本、请求索引、状态
  initial_state.json            # 评估起始缓存与策略状态
  final_state.json              # 可恢复状态
  detail/
    index.json                  # 封存状态、分片位置、计数、SHA-256
    access-000000.npy ...        # 紧凑的逐 request/token/layer 标注
    cache_events.jsonl.gz        # 初始化/预热/评估的加载和淘汰等事件
  summary.json
```

access 分片是结构化 NumPy 数组，字段为 `probe_tick, step_id, request_index, token_index, layer_id, expert_ids[top_k], hit_mask[top_k], hit_count`；request_index 查 manifest 中相应 phase 的请求表。所有标注可导出为符合 `AccessRecord` 的 JSONL。`probe_tick` 对应判定时已完成的缓存事件逻辑序号，便于从事件重建该次缓存快照；事件的 `context_layer_id` 标明触发 hook 的层，`layer_id` 标明被操作的专家层。数据文件保存自身的 expert IDs，后处理无需访问原始数据集。

Summary 的 `hit_rate` 对所有 evaluation 判断平等聚合，不按 request 或 layer 先求平均。输出 p10/p50/p90/p99/mean/median，数值范围 [0,1]，按命中率从高到低排序计算分位数，使用 linear 插值：P10 对应升序 Q90，P50 对应 Q50，P90 对应 Q10，P99 对应 Q1；结果标记 `percentile_order="descending"`，满足 P10 ≥ P50 ≥ P90 ≥ P99。P50 与 median 相同。`transfer` 按 transfer_complete 累计，避免 start/complete 双计数，报告各阶段搬运及 batch 去重数量。

LFU 使用驻留期间的频次，淘汰后清零；同频次按最近访问顺序淘汰，再按专家键确定并列顺序。当前 batch 的全部路由出现次数都计入频次，缺失专家按首次出现顺序处理。

## Strata

已实现同步 adaptive residency 策略。配置和与真实引擎的差别见 [docs/strata.md](docs/strata.md)。先生成独立 profile，再运行相同条件的两个 smoke：

```bash
expertcache profile dataset/router_results_sample100/benchmark.json --max-requests 20 --output runs/profile-train20.json
expertcache run configs/strata-smoke.json --output runs/strata-smoke
expertcache run configs/lfu-profile-smoke.json --output runs/lfu-profile-smoke
```

完整留出集评估用 `configs/strata-eval.json` 和 `configs/lfu-eval.json`。Strata 保持初始化后的层内槽位数，缺失专家临时加载，周期性适应换入单独计费。
