"""Time combinations of techniques served by vLLM: throughput, speedup over plain vLLM and acceptance, per batch size.

    pip install "economize[vllm]"
    python examples/benchmark_vllm.py --model Qwen/Qwen2.5-0.5B --out bench-vllm \\
        --drafter eagle3=runs/eagle3/final \\
        --recipe plain int4 fp8kv int4+fp8kv eagle3 eagle3+int4 eagle3+int4+fp8kv --batch 1 4 16 64 --k 4 6

A recipe joins components with ``+``: the weight and cache components of :data:`COMPONENTS` and the drafters named
with ``--drafter NAME=CHECKPOINT`` (any economize drafter checkpoint: EAGLE-3, Medusa, early exit, ...). ``plain`` is
the model alone. Each recipe is exported once with :func:`economize.backends.vllm.serving.to_vllm` (the drafter on the
compressed target), then every draft length ``--k`` gets its own engine, which decodes the prompts in static waves of
``B`` rows for every batch size ``B``: greedy, ``--max-new`` tokens per row with ``ignore_eos``, so each wave holds
``B`` running sequences. Steps run in subprocesses (vLLM owns the GPUs, and a failed configuration must not end the
run); finished configurations are skipped when the command is repeated. ``--export-only`` writes the exports and
stops.

Output ``OUT/vllm.json``: ``configs`` (per ``recipe_k<k>``: tok/s, wave times, vLLM's speculative counters and the
token ids per batch size) and ``rows`` (per configuration and batch size: tok/s, speedup over ``plain`` at the same
batch size, mean accepted length ``1 + accepted / drafts``, acceptance per draft position, and ``match``, the mean
fraction of each row equal to plain's tokens up to the first difference; bf16 kernels of a verification step and of
a one-token step round differently, so lossless speculative decoding parts from plain after some tokens).
Timing needs a CUDA device; ``--model`` must be in the Hugging Face cache or on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips to her "
    "friends in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Photosynthesis is the process by which",
    "In machine learning, overfitting means",
    "The French Revolution began in 1789 when",
    "Here is a recipe for a simple tomato soup:\n1.",
    "import numpy as np\n\ndef softmax(x):\n",
    "The three laws of thermodynamics state that",
]


def _int4() -> Any:
    from economize.techniques.quant.config import QuantConfig

    return QuantConfig(format="int4", group_size=128, backend="compressed-tensors")


def _int8() -> Any:
    from economize.techniques.quant.config import QuantConfig

    return QuantConfig(format="int8", backend="compressed-tensors")


def _fp8() -> Any:
    from economize.techniques.quant.config import QuantConfig

    return QuantConfig(format="fp8", backend="compressed-tensors")


def _fp8kv() -> Any:
    from economize.techniques.kv_quant.fp8 import FP8KVConfig

    return FP8KVConfig()


#: Recipe components without a checkpoint: name -> technique config (served natively by vLLM).
COMPONENTS: dict[str, Callable[[], Any]] = {
    "int4": _int4,  # compressed-tensors W4A16, groups of 128 (Marlin / Machete kernels)
    "int8": _int8,  # compressed-tensors W8A16
    "fp8": _fp8,  # compressed-tensors fp8 weights
    "fp8kv": _fp8kv,  # e4m3 KV cache, unit scales
}


# ---------------------------------------------------------------- step 1: export one recipe
def do_export(a: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    import economize
    from economize.backends.vllm.serving import VllmPlan, to_vllm
    from economize.core.serialization import attach, read_checkpoint

    if a.recipe == "plain":  # the model as it is: nothing to load
        VllmPlan(model=a.model, engine={"dtype": a.dtype}).save(Path(a.export_out).mkdir(parents=True) or a.export_out)
        return
    parts = a.recipe.split("+")
    drafters = dict(spec.split("=", 1) for spec in a.drafter)
    runs = [drafters[p] for p in parts if p in drafters]
    configs = [COMPONENTS[p]() for p in parts if p in COMPONENTS]
    device = a.export_device if a.export_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=device)
    if len(runs) > 1:
        raise SystemExit(f"recipe {a.recipe!r} has {len(runs)} drafters; vLLM runs one")
    if runs:
        checkpoint = read_checkpoint(runs[0])
        checkpoint.configs = configs + list(checkpoint.configs)
        model = attach(base, checkpoint, check_teacher=False)
    else:
        model = economize.apply(base, *configs) if configs else base
    plan = to_vllm(model, a.export_out, num_speculative_tokens=max(a.k_all), sampling=False, dtype=a.dtype)
    print(plan.describe(), flush=True)


# ---------------------------------------------------------------- step 2: one engine, every batch size
def spec_counters(llm: Any) -> dict[str, Any]:
    """vLLM's speculative-decoding counters (``spec_decode_*``), cumulative."""
    out: dict[str, Any] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name:
            key = metric.name.split(":")[-1]
            out[key] = list(metric.values) if hasattr(metric, "values") else metric.value
    return out


def _delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in after.items():
        if isinstance(value, list):
            old = before.get(key) or [0] * len(value)
            out[key] = [x - y for x, y in zip(value, old)]
        else:
            out[key] = value - before.get(key, 0)
    return out


def load_prompts(path: str, model: str) -> tuple[list[list[int]], list[str]]:
    """Token ids and task names: ``path`` is JSON ``{"prompts": [{"ids" | "text": ..., "task": ...}]}``, or empty."""
    from transformers import AutoTokenizer

    items = json.loads(Path(path).read_text())["prompts"] if path else [{"text": p} for p in PROMPTS]
    tokenizer = None
    ids = []
    for item in items:
        if "ids" in item:
            ids.append(list(item["ids"]))
            continue
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
        ids.append(tokenizer(item["text"]).input_ids)
    return ids, [str(item.get("task", "")) for item in items]


def do_bench(a: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from economize.backends.vllm.serving import VllmPlan

    plan = VllmPlan.load(a.export_out)
    prompts, tasks = load_prompts(a.prompts, a.model)
    overrides: dict[str, Any] = dict(
        tensor_parallel_size=a.tp, max_model_len=a.max_model_len, seed=0, disable_log_stats=False,
        enable_prefix_caching=False, max_num_seqs=max(a.batch), gpu_memory_utilization=a.gpu_util,
    )
    if a.multimodal_off:
        overrides["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    if plan.speculative is not None:
        overrides["speculative_config"] = {**plan.speculative, "num_speculative_tokens": a.k}
    if a.attention_backend:
        overrides["attention_config"] = {"backend": a.attention_backend}
    started = time.perf_counter()
    llm = plan.llm(**overrides)
    load_s = time.perf_counter() - started
    attention = plan.llm_kwargs(**overrides).get("attention_config")
    params = SamplingParams(temperature=0, max_tokens=a.max_new, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=prompts[0])], SamplingParams(temperature=0, max_tokens=16),
                 use_tqdm=False)  # warm-up
    result: dict[str, Any] = {"plan": plan.to_dict(), "k": a.k if plan.speculative else 0, "tp": a.tp,
                              "attention_config": attention, "load_s": load_s, "tasks": tasks, "batches": {}}
    for b in a.batch:
        n = -(-max(a.waves * b, a.min_rows) // b) * b
        rows = [prompts[i % len(prompts)] for i in range(n)]
        llm.generate([TokensPrompt(prompt_token_ids=rows[0])], SamplingParams(temperature=0, max_tokens=8),
                     use_tqdm=False)
        before = spec_counters(llm)
        waves, ids = [], []
        started = time.perf_counter()
        for i in range(0, n, b):
            wave = time.perf_counter()
            outs = llm.generate([TokensPrompt(prompt_token_ids=r) for r in rows[i : i + b]], params, use_tqdm=False)
            waves.append(time.perf_counter() - wave)
            ids += [list(o.outputs[0].token_ids) for o in outs]
        seconds = time.perf_counter() - started
        spec = _delta(spec_counters(llm), before)
        tokens = sum(map(len, ids))
        result["batches"][str(b)] = {"rows": n, "seconds": seconds, "wave_seconds": waves, "tokens": tokens,
                                     "tok_s": tokens / seconds, "spec": spec, "ids": ids,
                                     "prompt_idx": [i % len(prompts) for i in range(n)]}
        print(f"{a.recipe} k{a.k} B={b}: {tokens / seconds:.1f} tok/s "
              f"{ {k: v for k, v in spec.items() if not isinstance(v, list)} }", flush=True)
    Path(a.bench_out).write_text(json.dumps(result))


# ---------------------------------------------------------------- driver and summary
#: vLLM aborts start-up when another process (often the previous engine, still exiting) frees GPU memory while it
#: profiles; such a step is retried.
TRANSIENT = "Error in memory profiling"


def run_step(argv: list[str], log: Path, env: dict[str, str], timeout: float) -> tuple[int, str]:
    """Run this script with ``argv`` in a subprocess; its return code and the tail of its log."""
    with open(log, "w") as fh:
        try:
            rc = subprocess.run([sys.executable, str(HERE), *argv], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, "\n".join(log.read_text(errors="replace").splitlines()[-40:])


def prefix_match(a: list[int], b: list[int]) -> float:
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return n / max(len(b), 1)


def summarize(configs: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (configuration, batch size), relative to ``plain`` at the same batch size."""
    plain = configs.get("plain", {}).get("batches", {})
    rows = []
    for tag, config in configs.items():
        for b, r in config.get("batches", {}).items():
            s = {k.removeprefix("spec_decode_"): v for k, v in r["spec"].items()}
            drafts, accepted = s.get("num_drafts") or 0, s.get("num_accepted_tokens") or 0
            row: dict[str, Any] = {
                "config": tag, "recipe": tag.rsplit("_k", 1)[0] if config.get("k") else tag, "k": config.get("k"),
                "batch": int(b), "tok_s": r["tok_s"], "tokens": r["tokens"],
                "attention": (config.get("attention_config") or {}).get("backend"),
                "acc_len": 1 + accepted / drafts if drafts else None,
                "accept_rate": accepted / s["num_draft_tokens"] if s.get("num_draft_tokens") else None,
                "accept_per_pos": [x / drafts for x in s.get("num_accepted_tokens_per_pos") or []] if drafts else None,
            }
            if b in plain:
                row["speedup"] = r["tok_s"] / plain[b]["tok_s"]
                matches = [prefix_match(x, y) for x, y in zip(r["ids"], plain[b]["ids"])]
                row["match"] = sum(matches) / max(len(matches), 1)
            rows.append(row)
    return rows


def _environment() -> dict[str, str]:
    try:
        import torch
        import vllm

        import economize

        return {"vllm": vllm.__version__, "torch": torch.__version__, "economize": economize.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    except Exception as error:
        return {"error": repr(error)}


def drive(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    known = set(COMPONENTS) | {spec.split("=", 1)[0] for spec in a.drafter}
    for recipe in a.recipe:
        unknown = [p for p in recipe.split("+") if p not in known and recipe != "plain"]
        if unknown:
            raise SystemExit(f"recipe {recipe!r}: unknown components {unknown} (known: {sorted(known)})")
    env = dict(os.environ)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    result_path = out / "vllm.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"configs": {}}
    result["args"] = vars(a)
    common = ["--model", a.model, "--dtype", a.dtype]
    for recipe in a.recipe:
        drafted = any(p not in COMPONENTS for p in recipe.split("+")) and recipe != "plain"
        tags = [(f"{recipe}_k{k}", k) for k in a.k] if drafted else [(recipe, 0)]
        if all("batches" in result["configs"].get(tag, {}) for tag, _ in tags):
            continue
        export = out / "exports" / recipe
        if not (export / "economize_vllm.json").exists():
            argv = ["--step", "export", *common, "--recipe", recipe, "--export-out", str(export),
                    "--export-device", a.export_device, "--k", *map(str, a.k),
                    *[x for spec in a.drafter for x in ("--drafter", spec)]]
            export_env = env | ({"CUDA_VISIBLE_DEVICES": ""} if a.export_device == "cpu" else {})
            rc, tail = run_step(argv, out / f"export_{recipe}.log", export_env, a.timeout)
            if rc != 0:
                result["configs"][f"export/{recipe}"] = {"error": tail, "rc": rc}
                print("export failed", recipe, tail[-2000:], flush=True)
                result_path.write_text(json.dumps(result))
                continue
        if a.export_only:
            print("exported", recipe, "->", export, flush=True)
            continue
        for tag, k in tags:
            if "batches" in result["configs"].get(tag, {}):
                continue
            bench_out = out / f"cfg_{tag}.json"
            argv = ["--step", "bench", *common, "--recipe", recipe, "--export-out", str(export), "--k", str(k or 1),
                    "--prompts", a.prompts, "--tp", str(a.tp), "--batch", *map(str, a.batch), "--max-new",
                    str(a.max_new), "--max-model-len", str(a.max_model_len), "--waves", str(a.waves), "--min-rows",
                    str(a.min_rows), "--gpu-util", str(a.gpu_util), "--bench-out", str(bench_out)]
            argv += ["--attention-backend", a.attention_backend] if a.attention_backend else []
            argv += ["--multimodal-off"] if a.multimodal_off else []
            for attempt in range(a.tries):
                rc, tail = run_step(argv, out / f"cfg_{tag}.log", env, a.timeout)
                if rc == 0 or TRANSIENT not in tail or attempt + 1 == a.tries:
                    break
                print("retrying", tag, "(GPU memory changed during vLLM's profiling)", flush=True)
                time.sleep(30)
            if rc == 0 and bench_out.exists():
                result["configs"][tag] = json.loads(bench_out.read_text())
            else:
                result["configs"][tag] = {"error": tail, "rc": rc}
                print("FAILED", tag, rc, tail[-1500:], flush=True)
            result["rows"] = summarize(result["configs"])
            result_path.write_text(json.dumps(result))
        if not a.keep_exports and (export / "model").exists():
            shutil.rmtree(export / "model")  # the compressed target (the plan and the drafter stay)
    result["rows"] = summarize(result["configs"])
    result["env"] = _environment()
    result_path.write_text(json.dumps(result))
    if a.export_only:
        return
    for row in result["rows"]:
        print(json.dumps({k: v for k, v in row.items() if k != "accept_per_pos"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", default="drive", choices=["drive", "export", "bench"], help=argparse.SUPPRESS)
    parser.add_argument("--out", default="bench-vllm")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--recipe", nargs="+", default=["plain", "int4", "fp8kv", "int4+fp8kv"],
                        help="components joined with '+' (see COMPONENTS and --drafter), or 'plain'")
    parser.add_argument("--drafter", action="append", default=[], metavar="NAME=CHECKPOINT",
                        help="a drafter checkpoint usable as a recipe component (repeatable)")
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--k", nargs="+", type=int, default=[4], help="draft lengths (one engine each)")
    parser.add_argument("--prompts", default="", help='JSON {"prompts": [{"ids" | "text": ..., "task": ...}]}')
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    parser.add_argument("--waves", type=int, default=2, help="timed waves per batch size (at least --min-rows rows)")
    parser.add_argument("--min-rows", type=int, default=16)
    parser.add_argument("--gpu-util", type=float, default=0.92)
    parser.add_argument("--attention-backend", default="", help="vLLM attention backend, e.g. TRITON_ATTN")
    parser.add_argument("--multimodal-off", action="store_true", help="no image / audio / video inputs (Gemma-4)")
    parser.add_argument("--export-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--keep-exports", action="store_true", help="keep the compressed targets after timing")
    parser.add_argument("--export-only", action="store_true", help="write the vLLM exports and stop (no GPU needed)")
    parser.add_argument("--timeout", type=float, default=5400, help="seconds per step")
    parser.add_argument("--tries", type=int, default=3, help="attempts per bench step after a transient failure")
    # one step (internal)
    parser.add_argument("--export-out", default="", help=argparse.SUPPRESS)
    parser.add_argument("--bench-out", default="", help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.step == "export":
        a.recipe, a.k_all = a.recipe[0], a.k
        return do_export(a)
    if a.step == "bench":
        a.recipe, a.k = a.recipe[0], a.k[0]
        return do_bench(a)
    return drive(a)


if __name__ == "__main__":
    main()
