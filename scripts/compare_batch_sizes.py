"""Compare capacity sweeps while checking all non-batch inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from expertcache.contracts import RunManifest
from expertcache.data import digest_file, write_json


def load(path):
    return json.loads(Path(path).read_text())


def normalized_config(manifest):
    config = manifest.config.model_dump(mode="json")
    config["dataset"] = manifest.dataset_fingerprint
    config["initialization"]["profile_path"] = manifest.inputs["profile_sha256"]
    config["scheduler"].pop("batch_size")
    return config


def metrics(entry):
    traffic = entry["transfer"]
    evaluation = traffic["phases"]["evaluation"]
    manifest = RunManifest.model_validate_json((Path(entry["run_dir"]) / "manifest.json").read_text())
    tokens = sum(min(r.token_count, manifest.config.scheduler.max_tokens or r.token_count)
                 for r in manifest.requests["evaluation"])
    progress = Path(entry["run_dir"]).with_suffix(".progress.json")
    return dict(hit_rate=entry["hit_rate"], evaluation=evaluation,
                evaluation_units_per_output_token=evaluation["units"] / tokens,
                miss_occurrences=traffic["evaluation_miss_occurrences"],
                unique_miss_demands=traffic["evaluation_unique_miss_demands"],
                coalesced_misses=traffic["evaluation_coalesced_miss_occurrences"],
                initial_units=traffic["phases"]["initialization"]["units"],
                policy_metadata=manifest.policy_metadata,
                elapsed_seconds=load(progress).get("elapsed_seconds") if progress.exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--verification", required=True, help="candidate sweep's full audit output")
    parser.add_argument("--output", required=True, help="output stem, without .json/.md")
    args = parser.parse_args()
    baseline, candidate, verification = map(load, (args.baseline, args.candidate, args.verification))
    if "report_sha256" in verification:
        assert verification["report_sha256"] == digest_file(args.candidate)
    before_bs, after_bs = baseline["experiment"]["batch_size"], candidate["experiment"]["batch_size"]
    assert before_bs != after_bs
    before = {(e["policy"], e["cache_percent"]): e for e in baseline["results"]}
    after = {(e["policy"], e["cache_percent"]): e for e in candidate["results"]}
    assert len(before) == len(baseline["results"]) and len(after) == len(candidate["results"])
    checked = {(e["policy"], e["cache_percent"]): e for e in verification["checks"]}
    assert set(checked) == set(after) and verification["verified_runs"] == len(after)
    rows = []
    for key, new in sorted(after.items()):
        old = before[key]
        old_dir, new_dir = Path(old["run_dir"]), Path(new["run_dir"])
        manifests = [RunManifest.model_validate_json((p / "manifest.json").read_text()) for p in (old_dir, new_dir)]
        for entry, directory, manifest, batch in zip((old, new), (old_dir, new_dir), manifests, (before_bs, after_bs)):
            assert manifest.status == "complete" and manifest.config.scheduler.batch_size == batch
            assert digest_file(directory / "manifest.json") == entry["manifest_sha256"]
            assert digest_file(directory / "summary.json") == entry["summary_sha256"]
            summary = load(directory / "summary.json")
            assert summary["hit_rate"] == entry["hit_rate"] and summary["transfer"] == entry["transfer"]
        assert normalized_config(manifests[0]) == normalized_config(manifests[1])
        assert manifests[0].requests == manifests[1].requests
        assert manifests[0].inputs["policy_implementation"]["source_sha256"] == manifests[1].inputs["policy_implementation"]["source_sha256"]
        initial = [load(p / "initial_state.json") for p in (old_dir, new_dir)]
        assert initial[0] == initial[1], f"Different initial state: {key}"
        a, b = metrics(new), metrics(old)
        assert checked[key]['batch_size'] == after_bs
        assert checked[key]["annotated_decisions"] == a["hit_rate"]["count"]
        assert abs(checked[key]["mean_independently_recomputed"] - a["hit_rate"]["mean"]) < 1e-12
        rows.append(dict(policy=key[0], cache_percent=key[1], capacity_units=new["capacity_units"],
                         candidate=a, baseline=b,
                         hit_rate_delta_pp=(a["hit_rate"]["mean"] - b["hit_rate"]["mean"]) * 100,
                         evaluation_units_delta_percent=(a["evaluation"]["units"] / b["evaluation"]["units"] - 1) * 100,
                         same_config_except_batch=True, same_initial_state=True, same_policy_source=True,
                         baseline_run=old["run_dir"], candidate_run=new["run_dir"]))
    report = dict(schema_version="1.0", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  candidate_batch_size=after_bs, baseline_batch_size=before_bs,
                  inputs={name:dict(path=str(Path(p).resolve()),sha256=digest_file(p))
                          for name,p in (("baseline",args.baseline),("candidate",args.candidate),("verification",args.verification))},
                  candidate_sweep_elapsed_seconds=candidate["elapsed_seconds"],
                  comparison_scope="Same traces, request order, initialization, capacity and policy; only scheduler batch size changes",
                  comparisons=rows)
    stem = Path(args.output)
    write_json(stem.with_suffix(".json"), report)
    lines = [f"# Batch size 对比：BS={after_bs} 与 BS={before_bs}", "",
             "沿用前 20 请求 profile，评估其余 80 请求 × 512 token × 40 有效层；top-k=6。每专家 1 unit，workspace=1 unit。",
             f"每组 1,638,400 条逐层标注；新增 BS={after_bs} 共 {len(rows)} 组、{verification['verified_decisions']:,} 条，全部完成独立核验。",
             "逐对核验配置（仅 batch size 不同）、请求顺序、完整初始缓存与策略状态、数据/profile 指纹和策略源码；历史结果保持原样。", "",
             "## 平均命中率", "",
             f"| 缓存 | 策略 | BS={after_bs} | BS={before_bs} | 差值（百分点） |",
             "|---|---|---:|---:|---:|"]
    ordered = sorted(rows, key=lambda r:(r['cache_percent'], {'strata':0,'belady':1}.get(r['policy'],2)))
    for r in ordered:
        a,b=r['candidate'],r['baseline']
        lines.append(f"| {r['cache_percent']}% | {r['policy']} | {a['hit_rate']['mean']:.2%} | {b['hit_rate']['mean']:.2%} | {r['hit_rate_delta_pp']:+.2f} |")
    lines += ["", "## Evaluation 权重搬运量", "",
              "包含 demand 与 adaptation 等全部 evaluation 加载，初始化单列。相同输出 token 数下直接比较总量；差值为候选相对基线。", "",
              f"| 缓存 | 策略 | BS={after_bs} units | BS={before_bs} units | 变化 | BS={after_bs} units/token | BS={before_bs} units/token |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in ordered:
        a,b=r['candidate'],r['baseline']
        lines.append(f"| {r['cache_percent']}% | {r['policy']} | {a['evaluation']['units']:,} | {b['evaluation']['units']:,} | {r['evaluation_units_delta_percent']:+.2f}% | {a['evaluation_units_per_output_token']:.2f} | {b['evaluation_units_per_output_token']:.2f} |")
    lines += ["", "## 去重与适应性换入", "",
              f"| 缓存 | 策略 | BS={after_bs} demand | BS={before_bs} demand | BS={after_bs} adaptation | BS={before_bs} adaptation | BS={before_bs} 合并 miss 数 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in ordered:
        a,b=r['candidate'],r['baseline']
        lines.append(f"| {r['cache_percent']}% | {r['policy']} | {a['evaluation']['demand_units']:,} | {b['evaluation']['demand_units']:,} | {a['evaluation']['adaptation_units']:,} | {b['evaluation']['adaptation_units']:,} | {b['coalesced_misses']:,} |")
    lines += ["", "## 比较口径", "",
              "- BS=1 顺序完成每个请求；BS>1 交错推进多个请求。缓存跨请求保留，因此 batch size 同时改变全局访问顺序与局部性。",
              "- Strata 参数保持不变：每 4 个 decode step 调整一次，decay=0.7、min_count=2、min_gain=1.5、max_swaps=96。没有按总 token 数重新缩放策略时钟；具体调整次数保留在 JSON 的 policy_metadata 中。",
              "- BS>1 同层重复 miss 可合并为一次 demand 加载；命中率仍按每个请求的 top-k 分别计数。不能只凭命中率推算权重搬运量。",
              "- Belady 针对各自调度后的访问序列，优化等大小全局池的去重需求加载，允许 bypass、固定初始驻留。BS>1 的 occurrence 加权命中率不保证最优。",
              "- 结果是离线 cache 命中与权重搬运统计，不是实际模型吞吐、TPOT 或 MTP 加速。", "",
              "## 耗时与核验", "",
              f"本轮 BS={after_bs} 扫描墙钟耗时 **{candidate['elapsed_seconds']:.2f} 秒**，独立核验部分与扫描并行。各组进程耗时在配套 JSON 中保留，包含数据校验、初始化、回放和 summary；并行运行下不能视为独占机器的算法性能排名。",
              "核验覆盖全部路由、记录顺序、覆盖范围、hit_mask/hit_count、均值与降序分位数、初始化和报告校验和；Belady 额外从缓存事件重建每个快照并检查每一步下一次使用的保留/排除边界。", "",
              "配套 JSON 保留每组完整命中率分位数、需求/适应性搬运、batch 合并量、来源指纹和配对检查结果。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"compared_runs":len(rows),"report":str(stem.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
