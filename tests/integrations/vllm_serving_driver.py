"""Driver of the vLLM serving GPU tests and benchmarks, run in a subprocess (vLLM owns the GPU).

    python vllm_serving_driver.py RECIPE OUT.json [--model M] [--dtype D] [--k K] [--max-tokens N] [--hf]
                                  [--teacher] [--single] [--ieee-attention] [--batch B ...] [--run RUN]
                                  [--workdir DIR]

A recipe builds an EfficientModel on ``--model`` (:data:`RECIPES`), :func:`~economize.backends.vllm.serving.to_vllm`
writes it to ``--workdir/RECIPE``, and vLLM decodes :data:`PROMPTS` greedily with the plan. ``--hf`` also decodes
them with the economize HF runtime (the reference of the weight techniques). ``--batch`` times decoding at the given
batch sizes (``B`` prompts per ``generate`` call, the prompts repeated). The JSON holds the token ids, the first
generated token's log-probability, the :data:`TOP` most likely tokens at every position, vLLM's speculative counters
and the timings.

``--teacher`` feeds vLLM's continuations to the economize HF runtime (prefill phase over the prompt, then decode
phase one token at a time) and records, per position, the HF log-probability of vLLM's token below HF's best one
(``tf_regret``), its distance from vLLM's own log-probability of it (``tf_diff``), the same distance for the dense
model (``tf_dense_diff``) and HF's distance from itself with eager attention (``tf_noise_diff``). TEAL's thresholds
amplify rounding differences: an entry within rounding of its threshold flips, and the flip spreads through the
following layers and the cache, so two correct implementations drift apart after a few tokens. Its tests compare
the drift of vLLM from HF with HF's own drift under a change of attention kernel. ``--single`` decodes the prompts
one request at a time (no batch composition effects), ``--ieee-attention`` makes vLLM's Triton attention kernels
multiply float32 inputs in IEEE precision (Triton's default for float32 ``tl.dot`` is TF32).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

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

#: Number of most likely tokens recorded per position.
TOP = 5


def _calibration(tokenizer: Any) -> list[torch.Tensor]:
    text = " ".join(PROMPTS) * 8
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]
    return [ids[:, i : i + 128] for i in range(0, ids.shape[1] - 127, 128)]


def _datastore(base: Any, tokenizer: Any, args: argparse.Namespace) -> str:
    """A suffix-index datastore of every prompt followed by the base model's greedy continuation (built once)."""
    from economize.techniques.retrieval import SuffixIndex

    folder = Path(args.workdir or Path(args.out).with_suffix("")) / "datastore"
    if not folder.is_dir():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base.to(device)
        records = []
        for prompt in PROMPTS:
            tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = base.generate(tokens, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=0)
            records.append(out[0].tolist())
        base.to("cpu")
        torch.cuda.empty_cache()
        SuffixIndex.build(records, depth=64).save(folder)
    return str(folder)


def _perturbed(technique: torch.nn.Module, std: float) -> None:
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in technique.parameters():
            if parameter.requires_grad or parameter.dim() == 2:
                parameter.add_(torch.randn(parameter.shape, generator=generator).to(parameter) * std)


def _trained_jump(base: Any, args: argparse.Namespace, *extra: Any, calibration: Any = None) -> Any:
    """The trained drafter of ``--run`` on ``base``, with the ``extra`` technique configs attached beside it."""
    from economize.core.serialization import attach, read_checkpoint

    if not args.run:
        raise SystemExit("this recipe needs --run (a trained jump drafter of the model)")
    checkpoint = read_checkpoint(args.run)
    checkpoint.configs = list(extra) + list(checkpoint.configs)
    return attach(base, checkpoint, calibration=calibration, check_teacher=False)


def _recipes() -> dict[str, Callable[[Any, Any, argparse.Namespace], Any]]:
    import economize
    from economize.techniques.dflash import DFlashConfig
    from economize.techniques.draft_model.config import DraftModelConfig
    from economize.techniques.dynamic_depth import DmtdConfig, FreeConfig, HiddenTransferConfig
    from economize.techniques.eagle3 import Eagle3Config
    from economize.techniques.early_exit import EarlyExitConfig
    from economize.techniques.frspec import FRSpecConfig
    from economize.techniques.hydra import HydraConfig
    from economize.techniques.kangaroo import KangarooConfig
    from economize.techniques.knapspec import KnapSpecConfig, SkipPlan
    from economize.techniques.kv_evict.technique import KVEvictConfig
    from economize.techniques.kv_kivi_fused import FusedKIVIConfig
    from economize.techniques.kv_quant.fp8 import FP8KVConfig
    from economize.techniques.kv_quant.kivi import KIVIConfig
    from economize.techniques.layer_skip import LayerSkipConfig
    from economize.techniques.lookahead.config import LookaheadConfig
    from economize.techniques.lookup.config import LookupConfig
    from economize.techniques.lowrank.config import LowRankConfig
    from economize.techniques.medusa import MedusaConfig
    from economize.techniques.prune.config import DepthPruneConfig, PruneConfig
    from economize.techniques.quant.config import DraftQuantConfig, QuantConfig
    from economize.techniques.redrafter import RedrafterConfig
    from economize.techniques.rest.config import RestConfig
    from economize.techniques.sparsity.config import TealConfig
    from economize.techniques.suffix_decoding.config import SuffixDecodingConfig

    streaming = KVEvictConfig(method="streaming", sinks=4, recent=16)
    # the prompts and 64 new tokens fill 70-130 slots: groups of 16 and a residual of 32 quantize most of them
    kivi = KIVIConfig(bits=4, group_size=16, residual_length=32)
    kivi_fused = FusedKIVIConfig(bits=4, group_size=16, residual_length=32)

    def fused(base: Any, *configs: Any) -> Any:
        """``configs`` on ``base`` with economize's attention (the HF runtime of ``kivi_fused``)."""
        return economize.apply(fused_base(base), *configs)

    def fused_base(base: Any) -> Any:
        from economize.attention import install

        install()
        base.set_attn_implementation("economize")
        return base

    def eagle3(base: Any, tok: Any, args: argparse.Namespace, *extra: Any) -> Any:
        model = economize.apply(base, *extra, Eagle3Config())
        _perturbed(model.techniques["eagle3"], 0.02)
        return model

    def frspec(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        ids = tuple(range(base.config.vocab_size // 4))
        model = economize.apply(base, FRSpecConfig(drafter=Eagle3Config(), vocab_size=len(ids), token_ids=ids))
        _perturbed(model.techniques["frspec"].inner, 0.02)
        return model

    def dflash(types: tuple[str, ...]) -> Callable[[Any, Any, argparse.Namespace], Any]:
        def build(base: Any, tok: Any, args: argparse.Namespace) -> Any:
            config = DFlashConfig(num_layers=len(types), layer_types=types, block_size=args.k + 1, sliding_window=64,
                                  mask_token_id=int(tok.pad_token_id or 0))
            model = economize.apply(base, config)
            _perturbed(model.techniques["dflash"], 0.02)
            return model

        return build

    def medusa(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        model = economize.apply(base, MedusaConfig(num_heads=args.k, num_layers=1))
        _perturbed(model.techniques["medusa"], 0.02)
        return model

    def early_exit_lora(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        exit_layer = base.config.num_hidden_layers // 2
        model = economize.apply(base, EarlyExitConfig(exit_layer=exit_layer, head="readout", adaptation="lora",
                                                      lora_rank=8))
        _perturbed(model.techniques["early_exit"], 0.002)
        return model

    def hydra(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        model = economize.apply(base, HydraConfig(num_heads=args.k, num_layers=1))  # Hydra++: with the prefix layer
        _perturbed(model.techniques["hydra"], 0.02)
        return model

    def hydra_shared(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        model = economize.apply(base, HydraConfig(num_heads=args.k, num_layers=1, prefix=False, head="shared"))
        _perturbed(model.techniques["hydra"], 0.02)
        return model

    def redrafter(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        model = economize.apply(base, RedrafterConfig(beam_length=args.k + 1))
        _perturbed(model.techniques["redrafter"], 0.02)
        return model

    def redrafter_shared(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        model = economize.apply(base, RedrafterConfig(beam_length=args.k + 1, head="shared", num_layers=1))
        _perturbed(model.techniques["redrafter"], 0.02)
        return model

    def dmtd(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        # perturbed copies of the late layers: the drafts differ from the target, so verification has work to do
        model = economize.apply(base, DmtdConfig(decoding_layers=max(1, base.config.num_hidden_layers // 4),
                                                 cycle_length=args.k + 1, adaptation="full"))
        _perturbed(model.techniques["dmtd"], 0.002)
        return model

    def hidden_transfer(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        # k maps from the middle up, with a repeated layer (two steps read the same auxiliary state)
        L = base.config.num_hidden_layers
        layers = tuple(sorted(min(L - 1, L // 2 + step // 2) for step in range(args.k)))
        model = economize.apply(base, HiddenTransferConfig(transfer_layers=layers))
        _perturbed(model.techniques["hidden_transfer"], 0.02)
        return model

    def kangaroo(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        model = economize.apply(base, KangarooConfig(exit_layer=2))
        _perturbed(model.techniques["kangaroo"].adapter_net, 0.02)  # o_proj starts small: let the attention count
        return model

    def knapspec(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        # a fixed skip set (the search runs while HF decodes): attention-only and MLP-only layers, a hole, the tail
        L = base.config.num_hidden_layers
        model = economize.apply(base, KnapSpecConfig())
        model.techniques["knapspec"].plan = SkipPlan(attention=frozenset({2, 4, L}), mlp=frozenset({4, 5, L}),
                                                     draft_length=args.k)
        return model

    def ct(fmt: str) -> QuantConfig:
        return QuantConfig(format=fmt, group_size=128, backend="compressed-tensors")

    def teal(*extra: Any, **config: Any) -> Callable[[Any, Any, argparse.Namespace], Any]:
        def build(base: Any, tok: Any, args: argparse.Namespace) -> Any:
            options = {"sparsity": 0.5, "allocation": "uniform", **config}
            return economize.apply(base, *extra, TealConfig(**options), calibration=_calibration(tok))

        return build

    def combo(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        return _trained_jump(base, args, ct("int4"), FP8KVConfig())

    def combo_target(base: Any, tok: Any, args: argparse.Namespace) -> Any:
        return economize.apply(base, ct("int4"), FP8KVConfig())

    return {
        "plain": lambda base, tok, args: base,
        "lookup": lambda base, tok, args: economize.apply(base, LookupConfig(max_ngram=3, min_ngram=1)),
        "suffix": lambda base, tok, args: economize.apply(base, SuffixDecodingConfig(max_spec_tokens=4)),
        # economize's custom_class proposer (economize.backends.vllm.proposers)
        "lookup_latest": lambda base, tok, args: economize.apply(
            base, LookupConfig(max_ngram=3, min_ngram=1, match="latest")),
        "lookahead_pool": lambda base, tok, args: economize.apply(
            base, LookaheadConfig(ngram=4, pool_from_prompt=True)),
        "rest": lambda base, tok, args: economize.apply(
            base, RestConfig(datastore=_datastore(base, tok, args), max_span=8, max_depth=args.k)),
        "suffix_datastore": lambda base, tok, args: economize.apply(
            base, SuffixDecodingConfig(datastore=_datastore(base, tok, args), max_spec_tokens=4)),
        "eagle3": eagle3,
        "frspec": frspec,
        "medusa": medusa,
        "dflash": dflash(("sliding_attention",) * 2),
        "dflash_mixed": dflash(("sliding_attention", "full_attention")),  # vLLM's V2 model runner
        "draft_model": lambda base, tok, args: economize.apply(
            base, DraftModelConfig(model=args.draft or args.model, dtype=args.dtype)),
        "early_exit": lambda base, tok, args: economize.apply(
            base, EarlyExitConfig(exit_layer=base.config.num_hidden_layers // 2)),
        "early_exit_lora": early_exit_lora,
        # every other inner layer skipped (holes in the path), and the last layers skipped (r < L, identity readout)
        "layer_skip": lambda base, tok, args: economize.apply(
            base, LayerSkipConfig(skip_layers=tuple(range(3, base.config.num_hidden_layers - 1, 2)))),
        "layer_skip_tail": lambda base, tok, args: economize.apply(
            base, LayerSkipConfig(skip_layers=(2, *range(base.config.num_hidden_layers - 2,
                                                         base.config.num_hidden_layers + 1)))),
        "kangaroo": kangaroo,
        "knapspec": knapspec,
        "hydra": hydra,
        "hydra_shared": hydra_shared,
        "redrafter": redrafter,
        "redrafter_shared": redrafter_shared,
        "hidden_transfer": hidden_transfer,
        "dmtd": dmtd,
        "free": lambda base, tok, args: economize.apply(
            base, FreeConfig(exit_layer=base.config.num_hidden_layers // 2, head="shared")),
        "jump": lambda base, tok, args: _trained_jump(base, args),
        "jump_draft_int8": lambda base, tok, args: _trained_jump(base, args, DraftQuantConfig(format="int8")),
        "jump_kv_fp8": lambda base, tok, args: _trained_jump(base, args, FP8KVConfig()),
        "quant_fake_int4": lambda base, tok, args: economize.apply(base, QuantConfig(format="int4", group_size=128)),
        "quant_ct_int4": lambda base, tok, args: economize.apply(base, ct("int4")),
        # kernel backends: the export reads the codes back from the packed weights (GPU-only modules)
        "quant_marlin_int4": lambda base, tok, args: economize.apply(
            base.to("cuda"), QuantConfig(format="int4", group_size=128, backend="marlin")),
        "quant_machete_int4": lambda base, tok, args: economize.apply(
            base.to("cuda"), QuantConfig(format="int4", group_size=128, backend="machete")),
        "quant_ct_int8": lambda base, tok, args: economize.apply(base, ct("int8")),
        "quant_ct_fp8": lambda base, tok, args: economize.apply(base, ct("fp8")),
        # economize's linear method: factored maps (vLLM's fused maps hold factored and dense shards)
        "lowrank": lambda base, tok, args: economize.apply(base, LowRankConfig(compression=0.2),
                                                           calibration=_calibration(tok)),
        "wanda": lambda base, tok, args: economize.apply(base, PruneConfig(method="wanda", sparsity=0.3),
                                                         calibration=_calibration(tok)),
        "depth_prune": lambda base, tok, args: economize.apply(base, DepthPruneConfig(num_blocks=2),
                                                               calibration=_calibration(tok)),
        "kv_fp8": lambda base, tok, args: economize.apply(base, FP8KVConfig()),
        "kv_fp8_tensor": lambda base, tok, args: economize.apply(base, FP8KVConfig(scale="tensor"),
                                                                 calibration=_calibration(tok)),
        # StreamingLLM (economize's CUSTOM attention backend): 4 sinks and a 16-token window, so every prompt
        # decodes past the window and frees pages
        "streaming": lambda base, tok, args: economize.apply(base, streaming),
        "streaming_layers": lambda base, tok, args: economize.apply(
            base, KVEvictConfig(sinks=4, recent=16, layers=tuple(range(1, base.config.num_hidden_layers + 1, 2)))),
        "streaming_eagle3": lambda base, tok, args: eagle3(base, tok, args, streaming),
        "streaming_early_exit": lambda base, tok, args: economize.apply(
            base, streaming, EarlyExitConfig(exit_layer=base.config.num_hidden_layers // 2)),
        "streaming_kv_fp8": lambda base, tok, args: economize.apply(base, streaming, FP8KVConfig(scale="token")),
        "kivi": lambda base, tok, args: economize.apply(base, kivi),
        "kivi_2bit": lambda base, tok, args: economize.apply(base, KIVIConfig(bits=2, group_size=16,
                                                                              residual_length=32)),
        "kivi_layers": lambda base, tok, args: economize.apply(base, KIVIConfig(
            bits=4, group_size=16, residual_length=32, layers=tuple(range(1, base.config.num_hidden_layers, 2)))),
        "kivi_fused": lambda base, tok, args: fused(base, kivi_fused),
        "kivi_eagle3": lambda base, tok, args: eagle3(base, tok, args, kivi),
        "kivi_fused_eagle3": lambda base, tok, args: eagle3(fused_base(base), tok, args, kivi_fused),
        "kivi_fused_early_exit": lambda base, tok, args: fused(
            base, kivi_fused, EarlyExitConfig(exit_layer=base.config.num_hidden_layers // 2)),
        "kivi_fused_jump": lambda base, tok, args: _trained_jump(fused_base(base), args, kivi_fused),
        # TEAL (economize's linear method): dense prompts and thresholded decoding rows; float32 runs the masked
        # dense map, bfloat16 TEAL's sparse GEMV against the HF runtime with the same kernel
        "teal": teal(),
        "teal_triton": teal(backend="triton"),
        "teal_second_half": teal(prefill="second_half"),
        # TEAL on both factors of the factored maps (the inputs and the latents down(x))
        "lowrank_teal": teal(LowRankConfig(compression=0.2)),
        "lowrank_teal_triton": teal(LowRankConfig(compression=0.2), backend="triton"),
        "teal_lookup": teal(LookupConfig(max_ngram=3, min_ngram=1), prefill="all"),
        "combo": combo,
        "combo_target": combo_target,
    }


def _hf_greedy(model: Any, tokenizer: Any, max_tokens: int) -> tuple[list[list[int]], list[float]]:
    from economize.core.technique import Capability
    from economize.unwrap import base_model_of

    base = base_model_of(model)
    # KV-cache techniques act only inside economize's decoding loop; weight techniques edit the base in place
    # (and activation sparsity reads the phase that loop sets)
    kv = any(Capability.KV_CACHE in t.capabilities or Capability.ACTIVATIONS in t.capabilities
             for t in getattr(model, "techniques", {}).values())
    generator = model if kv else base
    ids, first = [], []
    for prompt in PROMPTS:
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(base.device)
        with torch.no_grad():
            out = generator.generate(tokens, max_new_tokens=max_tokens, min_new_tokens=max_tokens, do_sample=False,
                                output_scores=True, return_dict_in_generate=True, pad_token_id=0)
        ids.append(out.sequences[0, tokens.shape[1] :].tolist())
        first.append(float(torch.log_softmax(out.scores[0][0].float(), -1).max()))
    return ids, first


def _forced(base: Any, tokenizer: Any, ids: list[list[int]]) -> list[list[tuple[float, float]]]:
    """Per prompt and position: HF's log-probability of vLLM's token and HF's best log-probability there.

    The prompt runs in the ``"prefill"`` phase and every generated token in the ``"decode"`` phase.
    """
    from economize.runtime.phase import phase as scope

    rows = []
    for prompt, row in zip(PROMPTS, ids, strict=True):
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(base.device)
        scores = []
        with torch.no_grad():
            with scope("prefill"):
                out = base(tokens, use_cache=True)
            for token in row:
                logprobs = torch.log_softmax(out.logits[0, -1].double(), -1)
                scores.append((float(logprobs[token]), float(logprobs.max())))
                with scope("decode"):
                    out = base(torch.tensor([[token]], device=base.device), past_key_values=out.past_key_values,
                               use_cache=True)
        rows.append(scores)
    return rows


def _teacher_forced(model: Any, tokenizer: Any, ids: list[list[int]],
                    served: list[list[list[Any]]]) -> dict[str, list[list[float]]]:
    """vLLM's continuations scored by the economize HF runtime, per prompt and position (see ``--teacher``).

    ``tf_regret``: HF's best log-probability minus HF's log-probability of vLLM's token. ``tf_diff``: the distance
    between HF's and vLLM's log-probabilities of vLLM's token. ``tf_dense_diff``: the same distance from the dense
    model (TEAL's hooks removed). ``tf_noise_diff``: the distance between HF's log-probabilities with eager and with
    the default attention implementation, HF's own sensitivity to rounding along the same tokens.
    """
    from economize.unwrap import base_model_of

    base = base_model_of(model)
    lp = [[{int(t): v for t, v in position} for position in row] for row in served]
    from economize.techniques.sparsity import TealTechnique

    ref = _forced(base, tokenizer, ids)
    teal = [module for module in model.modules() if isinstance(module, TealTechnique)]
    for technique in teal:
        technique.detach()
    try:
        dense = _forced(base, tokenizer, ids)
    finally:
        for technique in teal:
            technique.install()
    implementation = base.config._attn_implementation
    base.set_attn_implementation("eager")
    try:
        eager = _forced(base, tokenizer, ids)
    finally:
        base.set_attn_implementation(implementation)

    def table(score: Callable[[int, int, int], float]) -> list[list[float]]:
        return [[score(i, j, token) for j, token in enumerate(row)] for i, row in enumerate(ids)]

    def gap(scores: list[list[tuple[float, float]]]) -> Callable[[int, int, int], float]:
        return lambda i, j, token: abs(scores[i][j][0] - lp[i][j][token]) if token in lp[i][j] else float("inf")

    return {"tf_regret": table(lambda i, j, token: ref[i][j][1] - ref[i][j][0]),
            "tf_diff": table(gap(ref)), "tf_dense_diff": table(gap(dense)),
            "tf_noise_diff": table(lambda i, j, token: abs(eager[i][j][0] - ref[i][j][0]))}


def _kv_scales(model: Any) -> list[list[float]]:
    """The k / v scales vLLM's attention layers hold after loading, in layer order (runs inside the worker)."""
    return [[float(module._k_scale_float), float(module._v_scale_float)]
            for _, module in model.named_modules() if hasattr(module, "_k_scale_float")]


#: Recipes whose modules hold CUDA kernel weights that cannot move to the CPU.
GPU_ONLY_RECIPES = {"quant_marlin_int4", "quant_machete_int4"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("out")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--draft", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--batch", type=int, nargs="*", default=[])
    parser.add_argument("--bench-tokens", type=int, default=256)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--attention-backend", default=None, help="e.g. TRITON_ATTN (vLLM attention_config.backend)")
    parser.add_argument("--kv-scales", action="store_true", help="record the KV scales vLLM loaded")
    parser.add_argument("--teacher", action="store_true", help="teacher-force vLLM's tokens through the HF runtime")
    parser.add_argument("--single", action="store_true", help="decode the prompts one request at a time")
    parser.add_argument("--ieee-attention", action="store_true", help="IEEE float32 dots in vLLM's Triton kernels")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size of the engine")
    args = parser.parse_args()
    if args.ieee_attention:  # read when Triton compiles a kernel, in the engine core too
        os.environ["TRITON_F32_DEFAULT"] = "ieee"

    import transformers

    from economize.backends.vllm import serving

    dtype = getattr(torch, args.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    base = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    model = _recipes()[args.recipe](base, tokenizer, args)
    result: dict[str, Any] = {"recipe": args.recipe, "dtype": args.dtype}
    if args.hf:
        if torch.cuda.is_available():
            model.to("cuda")
        result["hf_ids"], result["hf_first_logprob"] = _hf_greedy(model, tokenizer, args.max_tokens)
        if args.recipe not in GPU_ONLY_RECIPES:
            model.to("cpu")
        torch.cuda.empty_cache()
    workdir = Path(args.workdir or Path(args.out).with_suffix(""))
    plan = serving.to_vllm(model, workdir / args.recipe, num_speculative_tokens=args.k, sampling=False,
                           dtype=args.dtype)
    if plan.tokenizer is None and plan.model != args.model:
        plan.tokenizer = args.model
    result["plan"] = plan.to_dict()
    if not args.teacher:
        del model
    del base

    from vllm import SamplingParams

    extra: dict[str, Any] = {}
    if args.attention_backend:
        extra["attention_config"] = {"backend": args.attention_backend}
    if args.tp > 1:
        extra["tensor_parallel_size"] = args.tp
    if args.kv_scales:  # apply_model sends a function to the engine core
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = plan.llm(max_model_len=1024, seed=0, disable_log_stats=False, enforce_eager=args.eager,
                   gpu_memory_utilization=float(os.environ.get("ECONOMIZE_VLLM_GPU_UTIL", "0.4")), **extra)
    llm.generate(PROMPTS[:1], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warm-up
    if args.kv_scales:
        result["vllm_kv_scales"] = llm.apply_model(_kv_scales)[0]
    # with --hf, EOS is masked as HF's min_new_tokens masks it in the reference (ignore_eos alone still emits it)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True, logprobs=TOP,
                            min_tokens=args.max_tokens if args.hf else 0)
    if args.single:
        outputs = [llm.generate([prompt], params, use_tqdm=False)[0] for prompt in PROMPTS]
    else:
        outputs = llm.generate(PROMPTS, params, use_tqdm=False)
    result["ids"] = [list(output.outputs[0].token_ids) for output in outputs]
    # per prompt and position: the TOP most likely tokens and their log-probabilities (the near-tie check)
    result["top"] = [
        [sorted(([token, float(lp.logprob)] for token, lp in position.items()), key=lambda t: -t[1])[:TOP]
         for position in output.outputs[0].logprobs]
        for output in outputs
    ]
    result["first_logprob"] = [
        float(o.outputs[0].logprobs[0][o.outputs[0].token_ids[0]].logprob) for o in outputs
    ]
    if args.teacher:
        served = [[[[token, float(lp.logprob)] for token, lp in position.items()] for position in o.outputs[0].logprobs]
                  for o in outputs]
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        result.update(_teacher_forced(model, tokenizer, result["ids"], served))
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    timings = {}
    for batch in args.batch:
        prompts = (PROMPTS * (batch // len(PROMPTS) + 1))[:batch]
        bench = SamplingParams(temperature=0, max_tokens=args.bench_tokens, ignore_eos=True)
        llm.generate(prompts, bench, use_tqdm=False)
        runs = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            done = llm.generate(prompts, bench, use_tqdm=False)
            seconds = time.perf_counter() - start
            runs.append(sum(len(o.outputs[0].token_ids) for o in done) / seconds)
        timings[str(batch)] = {"tokens_per_second": sorted(runs)[1], "runs": runs}
    result["timings"] = timings
    counters: dict[str, Any] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name:
            counters[metric.name.split(":")[-1]] = list(metric.values) if hasattr(metric, "values") else metric.value
    result["spec"] = counters
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
