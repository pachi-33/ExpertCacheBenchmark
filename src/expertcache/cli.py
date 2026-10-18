import argparse
import json
import sys
from pathlib import Path
from .contracts import (RunConfig, DatasetManifest, AccessRecord, CacheEvent, RunManifest, DetailIndex,
                        Profile, SchedulerConfig, TokenPositionPoint, TokenPositionMetadata, TokenPositionSummary)
from .data import Dataset, import_router_dump, build_profile, write_json
from .detail import DetailReader
from .runner import Benchmark, load_config
from .scheduler import select_requests
from .summary import postprocess


def parse_layers(value):
    result = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.extend(range(start, end + 1))
        else: result.append(int(part))
    return tuple(result)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="expertcache")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="replay a JSON experiment config")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--quiet", action="store_true")
    validate = commands.add_parser("validate", help="validate manifest and every route array")
    validate.add_argument("dataset")
    imp = commands.add_parser("import-router-dump", help="add a native manifest to an existing router dump")
    imp.add_argument("directory")
    imp.add_argument("--valid-layers", required=True)
    imp.add_argument("--excluded-reason", default="unobserved routing; excluded from evaluation")
    imp.add_argument("--name", default="benchmark.json")
    profile = commands.add_parser("profile", help="count a declared training subset, without evaluation")
    profile.add_argument("dataset")
    profile.add_argument("--output", required=True)
    profile.add_argument("--max-requests", type=int)
    profile.add_argument("--request-offset", type=int, default=0)
    profile.add_argument("--max-tokens", type=int)
    post = commands.add_parser("postprocess", help="rebuild summary from sealed detail")
    post.add_argument("run_dir")
    selection = post.add_mutually_exclusive_group()
    selection.add_argument("--hook", action="append", default=None, help="trusted module:Class; replaces configured extras")
    selection.add_argument("--only-hook", action="append", default=None,
                           help="run only this trusted module:Class; requires --output")
    post.add_argument("--output", help="new summary JSON outside the source run; leaves source files unchanged")
    export = commands.add_parser("export-detail", help="export per-layer access annotations as JSONL")
    export.add_argument("run_dir")
    export.add_argument("--output", required=True)
    export.add_argument("--phase", choices=["evaluation", "warmup"], default="evaluation")
    schema = commands.add_parser("schemas", help="export versioned Pydantic JSON schemas")
    schema.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            progress = None if args.quiet else lambda phase, step: print(f"{phase}: {step} decode steps completed", file=sys.stderr)
            result = Benchmark(load_config(args.config)).run(args.output, progress)
        elif args.command == "validate":
            ds = Dataset(args.dataset)
            result = dict(status="valid", fingerprint=ds.fingerprint, requests=len(ds.spec.requests), valid_layers=ds.spec.valid_layers)
        elif args.command == "import-router-dump":
            path = Path(args.directory) / args.name
            spec = import_router_dump(args.directory, path, parse_layers(args.valid_layers), args.excluded_reason)
            result = dict(manifest=str(path), requests=len(spec.requests))
        elif args.command == "profile":
            ds = Dataset(args.dataset)
            config = SchedulerConfig(max_requests=args.max_requests, request_offset=args.request_offset, max_tokens=args.max_tokens)
            requests = select_requests(ds, config)
            if not requests: raise ValueError("profile selection is empty")
            path = Path(args.output)
            if path.exists(): raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = build_profile(ds, requests, args.max_tokens)
            write_json(path, value.model_dump(mode="json"))
            result = dict(profile=str(path), requests=len(requests), experts=len(value.entries))
        elif args.command == "postprocess":
            result = postprocess(args.run_dir, args.hook, only_hooks=args.only_hook, summary_output=args.output)
            if args.output is None:
                path = Path(args.run_dir) / "manifest.json"
                manifest = RunManifest.model_validate_json(path.read_text())
                write_json(path, manifest.model_copy(update={"status": "complete"}).model_dump(mode="json"))
        elif args.command == "export-detail":
            manifest = RunManifest.model_validate_json((Path(args.run_dir) / "manifest.json").read_text())
            reader = DetailReader(args.run_dir, manifest)
            count = 0
            with Path(args.output).open("x", encoding="utf-8") as f:
                for record in reader.records(args.phase):
                    f.write(record.model_dump_json() + "\n")
                    count += 1
            result = dict(records=count, output=args.output)
        else:
            path = Path(args.output)
            path.mkdir(parents=True, exist_ok=True)
            for cls in (RunConfig, DatasetManifest, AccessRecord, CacheEvent, RunManifest, DetailIndex, Profile,
                        TokenPositionPoint, TokenPositionMetadata, TokenPositionSummary):
                write_json(path / (cls.__name__ + ".schema.json"), cls.model_json_schema())
            result = dict(schemas=str(path))
    except (ValueError, OSError, TypeError, PermissionError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
