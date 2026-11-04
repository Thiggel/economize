"""Export a model with its techniques to ONNX Runtime (:func:`to_onnx`) and run the export (:func:`load_onnx`)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ...errors import ConfigError
from ..runtime_export import ExportBuilder, RuntimeExport, dense_state
from .graph import CACHE_INPUT, matmul_nbits, trace_onnx
from .runtime import ONNX

if TYPE_CHECKING:
    import numpy.typing as npt
    import onnxruntime

__all__ = ["DTYPES", "MODEL_FILE", "OnnxCausalLM", "check_onnx", "load_onnx", "plain_model", "to_onnx"]

MODEL_FILE = "model.onnx"
#: Graph float types by name.
DTYPES = {"float32": torch.float32, "float16": torch.float16}
#: Graphs above this size keep their tensors in ``model.onnx.data`` (protobuf files are limited to 2 GiB).
_EXTERNAL_BYTES = 1 << 30


def check_onnx(model: nn.Module) -> ExportBuilder:
    """Check that ONNX Runtime can run every technique of ``model`` (nothing is written).

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model.

    Returns
    -------
    ExportBuilder
        The plan (support level of every technique).

    Raises
    ------
    UnsupportedBackendError
        A technique has no ONNX Runtime path.
    """
    return ONNX.plan(model, None)


def plain_model(builder: ExportBuilder, dtype: torch.dtype = torch.float32) -> nn.Module:
    """The base model's plain architecture holding :func:`dense_state` (every technique multiplied out).

    Parameters
    ----------
    builder : ExportBuilder
        The export plan.
    dtype : torch.dtype
        The type of the floating-point weights.

    Returns
    -------
    nn.Module
        A new model of the base's class in eval mode (the base is not changed).

    Raises
    ------
    ConfigError
        The dense state does not fit the plain architecture.
    """
    base = builder.base
    with torch.device("meta"):
        plain = type(base)(getattr(base, "config"))  # noqa: B009 (nn.Module types attributes as tensors)
    missing, unexpected = plain.load_state_dict(dense_state(builder, dtype), strict=False, assign=True)
    for name, buffer in plain.named_buffers():
        if buffer.is_meta:  # non-persistent buffers (rotary frequencies) are not in the state dict
            owner, _, leaf = name.rpartition(".")
            plain.get_submodule(owner).register_buffer(leaf, base.get_buffer(name).detach().clone(), persistent=False)
    missing = [key for key in missing if plain.get_parameter(key).is_meta]
    if missing or unexpected:
        raise ConfigError(f"the dense weights do not fit {type(base).__name__}: missing {missing[:3]}, "
                          f"unexpected {unexpected[:3]}")
    return plain.eval()


def to_onnx(
    model: nn.Module,
    out_dir: str | Path,
    *,
    tokenizer: object | None = None,
    dtype: str = "float32",
) -> RuntimeExport:
    """Write ``model`` and its techniques as one ONNX graph with KV-cache inputs that ONNX Runtime runs.

    The graph is traced from the plain architecture with every technique multiplied out (``torch.onnx.export``,
    dynamo); the maps whose ``quant`` codes fit ``MatMulNBits`` keep them. Its inputs and outputs are listed in
    :mod:`economize.backends.onnx.graph`.

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face causal LM that ``torch.export`` traces.
    out_dir : str or Path
        The export folder; it receives ``model.onnx`` (and ``model.onnx.data`` above 1 GiB), the Hugging Face
        ``config.json``, the tokenizer files when ``tokenizer`` is given and the manifest ``economize_onnx.json``.
    tokenizer : PreTrainedTokenizerBase, optional
        Saved beside the graph.
    dtype : str
        ``"float32"`` (the default) or ``"float16"``: the graph's float type (activations, dense weights,
        MatMulNBits scales).

    Returns
    -------
    RuntimeExport
        The written files and the support level of every technique.

    Raises
    ------
    UnsupportedBackendError
        A technique has no ONNX Runtime path.
    ConfigError
        ``dtype`` is unknown, or the dense weights do not fit the base's class.
    """
    if dtype not in DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    builder = ONNX.plan(model, out_dir)  # refuses an unsupported technique before the optional import below
    import onnx

    folder = builder.path()
    folder.mkdir(parents=True, exist_ok=True)
    plain = plain_model(builder, DTYPES[dtype])
    proto = trace_onnx(plain)
    matmul_nbits(proto, builder.quantized)
    target = builder.path(MODEL_FILE)
    files = [target]
    external = sum(param.numel() * param.element_size() for param in plain.parameters()) > _EXTERNAL_BYTES
    if external:
        files.append(builder.path(MODEL_FILE + ".data"))
    onnx.save_model(proto, str(target), save_as_external_data=external, location=MODEL_FILE + ".data")
    getattr(plain, "config").save_pretrained(folder)  # noqa: B009 (nn.Module types attributes as tensors)
    files.append(builder.path("config.json"))
    if tokenizer is not None:
        files.extend(Path(name) for name in getattr(tokenizer, "save_pretrained")(folder))  # noqa: B009
    command = ["python", "-c", f"from economize.backends.onnx import load_onnx; load_onnx({str(folder)!r})"]
    return builder.finish(target, files, command)


class OnnxCausalLM:
    """An exported graph in an ONNX Runtime session, with the cache loop of greedy decoding.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        A session of a graph from :func:`to_onnx`.
    """

    def __init__(self, session: onnxruntime.InferenceSession) -> None:
        self.session = session
        inputs = {arg.name: arg for arg in session.get_inputs()}
        self.num_layers = sum(1 for name in inputs if name.endswith(".key"))
        first = inputs[CACHE_INPUT.format(0, "key")]
        self.kv_heads, self.head_dim = int(first.shape[1]), int(first.shape[3])
        self.dtype = np.float16 if "float16" in first.type else np.float32

    def empty_cache(self, batch: int) -> list[npt.NDArray[np.floating]]:
        """The cache of no tokens (``past = 0``) for ``batch`` sequences."""
        shape = (batch, self.kv_heads, 0, self.head_dim)
        return [np.zeros(shape, dtype=self.dtype) for _ in range(2 * self.num_layers)]

    def step(
        self,
        input_ids: npt.NDArray[np.integer],
        attention_mask: npt.NDArray[np.integer],
        cache: Sequence[npt.NDArray[np.floating]],
    ) -> tuple[npt.NDArray[np.floating], list[npt.NDArray[np.floating]]]:
        """Run the graph once.

        Parameters
        ----------
        input_ids : np.ndarray
            ``[batch, seq]`` new tokens.
        attention_mask : np.ndarray
            ``[batch, past + seq]`` (1 = attend).
        cache : Sequence[np.ndarray]
            The cache tensors in layer order, key before value.

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            The logits ``[batch, seq, vocab]`` and the cache after the call.
        """
        feeds: dict[str, object] = {"input_ids": input_ids.astype(np.int64),
                                    "attention_mask": attention_mask.astype(np.int64)}
        names = [CACHE_INPUT.format(layer, kind) for layer in range(self.num_layers) for kind in ("key", "value")]
        feeds.update(zip(names, cache, strict=True))
        logits, *present = self.session.run(None, feeds)
        return logits, present

    def generate(self, input_ids: npt.NDArray[np.integer], max_new_tokens: int) -> npt.NDArray[np.int64]:
        """Greedy decoding with the cache (unpadded prompts of equal length).

        Parameters
        ----------
        input_ids : np.ndarray
            ``[batch, prompt]`` token ids.
        max_new_tokens : int
            Tokens to generate.

        Returns
        -------
        np.ndarray
            ``[batch, prompt + max_new_tokens]`` token ids.
        """
        tokens = np.asarray(input_ids, dtype=np.int64)
        cache = self.empty_cache(tokens.shape[0])
        new = tokens
        for _ in range(max_new_tokens):
            mask = np.ones((tokens.shape[0], cache[0].shape[2] + new.shape[1]), dtype=np.int64)
            logits, cache = self.step(new, mask, cache)
            new = logits[:, -1].argmax(-1).astype(np.int64)[:, None]
            tokens = np.concatenate([tokens, new], axis=1)
        return tokens


def load_onnx(
    export: RuntimeExport | str | Path, *, providers: Sequence[str] = ("CPUExecutionProvider",)
) -> OnnxCausalLM:
    """Load an ONNX export in an ONNX Runtime session.

    Parameters
    ----------
    export : RuntimeExport or str or Path
        The export (or its folder).
    providers : Sequence[str]
        ONNX Runtime execution providers in order of preference.

    Returns
    -------
    OnnxCausalLM
        The session with the decoding loop.
    """
    import onnxruntime

    folder = export.path.parent if isinstance(export, RuntimeExport) else Path(export)
    return OnnxCausalLM(onnxruntime.InferenceSession(str(folder / MODEL_FILE), providers=list(providers)))
