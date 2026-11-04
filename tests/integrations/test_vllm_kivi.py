"""KIVI on vLLM (:mod:`economize.backends.vllm.kivi`): cell layout, reference, plans and backend hooks on CPU.

The kernels are checked on a GPU in ``test_vllm_kivi_gpu.py``; ``test_vllm_serving_gpu.py`` decodes with the backend
and compares it with the economize HF runtime.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("vllm")

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import custom_attention, plugin
from economize.backends.vllm.kivi import BACKEND_PATH, CONFIG_KEY
from economize.backends.vllm.kivi.backend import (
    EconomizeKIVIBackend,
    EconomizeKIVIImpl,
    EconomizeKIVIMetadataBuilder,
    layer_index,
)
from economize.backends.vllm.kivi.layout import (
    FLAG_DENSE,
    FLAG_OVERFLOW,
    FLAG_PACKED,
    SETTINGS,
    KIVISettings,
    cell_layout,
    read_cells,
    settings_of,
    write_cells,
)
from economize.backends.vllm.serving import check_servable, support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_evict.technique import KVEvictConfig
from economize.techniques.kv_kivi_fused import FusedKIVIConfig
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.kv_quant.kivi import fake_quantize
from helpers import make_tiny_model


def _tiny_head32(folder: Path, sliding: bool = False) -> Path:
    """A tiny Qwen2 with one KV head of 32 channels (vLLM's Triton attention needs head sizes of at least 32)."""
    model = make_tiny_model(seed=0)
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**{**model.config.to_dict(), "num_attention_heads": 2, "num_key_value_heads": 1})
    if sliding:
        config.layer_types = ["full_attention", "sliding_attention"] * 3
        config.sliding_window = 24
        config.use_sliding_window = True
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).float().eval().save_pretrained(folder)
    return folder


@pytest.fixture
def head32_dir(tmp_path: Path) -> Path:
    return _tiny_head32(tmp_path / "head32")


def _base(folder: Path, **config: Any) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, **config).eval()


def _fused_base(folder: Path) -> Any:
    from economize.attention import install

    install()
    return _base(folder, attn_implementation="economize")


# ---- settings and layout ----------------------------------------------------------------------------------------


def test_settings_validate_and_round_trip() -> None:
    settings = KIVISettings(bits=2, group_size=32, residual_length=128, layers=frozenset({1, 3}), num_layers=4,
                            fused=True)
    assert settings_of({CONFIG_KEY: settings.as_config()}) == settings
    assert settings_of({}) is None and settings_of(SimpleNamespace(additional_config=None)) is None
    assert settings.block_multiple == 32 and KIVISettings(4, 8, 8).block_multiple == 16
    assert [settings.covers(i) for i in range(6)] == [True, False, True, False, False, False]  # 4, 5: drafter
    assert all(KIVISettings(2, 32, 128, num_layers=2).covers(i) for i in range(2))
    for bad in ({"bits": 3}, {"group_size": 24, "residual_length": 48}, {"residual_length": 48}):
        with pytest.raises(ConfigError):
            KIVISettings(**{"bits": 2, "group_size": 32, "residual_length": 128, **bad})


def test_cell_layout_offsets() -> None:
    layout = cell_layout(64, torch.bfloat16, KIVISettings(2, 32, 128))
    assert (layout.off_v, layout.off_vs, layout.value_groups, layout.off_vz, layout.off_ka) == (128, 256, 2, 260, 264)
    assert (layout.key_aux, layout.off_kc, layout.off_vc, layout.off_flag, layout.size) == (4, 272, 288, 304, 320)
    assert layout.key_aux * layout.group_size >= 2 * layout.head_size
    small = cell_layout(16, torch.float32, KIVISettings(4, 32, 32))
    assert small.value_group == 16 and small.size % 16 == 0  # a value group wider than the head: one per token
    for head in (48, 8):
        with pytest.raises(ConfigError, match="power-of-two head size"):
            cell_layout(head, torch.bfloat16, KIVISettings(2, 32, 128))


def _grouped_keys(keys: torch.Tensor, g: int, bits: int) -> torch.Tensor:
    heads, tokens, dim = keys.shape
    return fake_quantize(keys.reshape(heads, tokens // g, g, dim), bits, dim=2).reshape(keys.shape)


def _grouped_values(values: torch.Tensor, vg: int, bits: int) -> torch.Tensor:
    heads, tokens, dim = values.shape
    return fake_quantize(values.reshape(heads, tokens, dim // vg, vg), bits, dim=3).reshape(values.shape)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("dim", "group"), [(32, 32), (64, 16), (128, 32), (16, 32)])
def test_cells_hold_economize_kivi_numbers(bits: int, dtype: torch.dtype, dim: int, group: int) -> None:
    """The cells' packed codes and dense entries both give HF KIVI's fake-quantized cache."""
    generator = torch.Generator().manual_seed(bits * dim + group)
    heads, tokens, kq, vq = 2, 3 * group + 5, 2 * group, 3 * group + 1
    keys = (torch.randn(heads, tokens, dim, generator=generator) * 3).to(dtype)
    values = torch.randn(heads, tokens, dim, generator=generator).to(dtype)
    layout = cell_layout(dim, dtype, KIVISettings(bits, group, group))
    cells = write_cells(keys, values, kq, vq, layout)
    want_k, want_v = keys.clone(), values.clone()
    want_k[:, :kq] = _grouped_keys(keys[:, :kq], group, bits)
    want_v[:, :vq] = _grouped_values(values[:, :vq], layout.value_group, bits)
    dense_k = cells[:, :, :layout.off_v].contiguous().view(dtype)
    dense_v = cells[:, :, layout.off_v:layout.off_vs].contiguous().view(dtype)
    assert torch.equal(dense_k, want_k) and torch.equal(dense_v, want_v)
    packed_k, packed_v = read_cells(cells, layout, dtype)
    assert torch.equal(packed_k, want_k) and torch.equal(packed_v, want_v)
    assert (cells[:, :kq, layout.off_flag] == FLAG_PACKED).all() and (cells[:, kq:, layout.off_flag] == 0).all()
    assert (cells[:, :vq, layout.off_flag + 1] == FLAG_PACKED).all()
    assert (cells[:, vq:, layout.off_flag + 1] == FLAG_DENSE).all()


def test_scales_beyond_fp16_keep_the_dense_entry() -> None:
    keys = torch.randn(1, 32, 32)
    keys[0, 3, 5] = 2e5  # channel 5 of the first key group: its scale and zero point leave the fp16 range
    values = keys.clone()
    layout = cell_layout(32, torch.float32, KIVISettings(2, 32, 32))
    cells = write_cells(keys, values, 32, 32, layout)
    assert (cells[0, :, layout.off_flag] == FLAG_OVERFLOW).all()
    v_flags = cells[0, :, layout.off_flag + 1]
    assert v_flags[3] == FLAG_OVERFLOW and (v_flags[torch.arange(32) != 3] == FLAG_PACKED).all()
    got_k, got_v = read_cells(cells, layout, torch.float32)
    assert torch.equal(got_k, _grouped_keys(keys, 32, 2)) and torch.equal(got_v, _grouped_values(values, 32, 2))
    assert torch.isfinite(got_k).all()


def test_partial_key_groups_are_refused() -> None:
    layout = cell_layout(32, torch.float32, KIVISettings(2, 32, 32))
    with pytest.raises(ConfigError, match="whole groups"):
        write_cells(torch.zeros(1, 40, 32), torch.zeros(1, 40, 32), 16, 0, layout)


# ---- plans ------------------------------------------------------------------------------------------------------


def test_plan_serves_kivi_on_the_custom_backend(head32_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(economize.apply(_base(head32_dir), KIVIConfig()), tmp_path / "out")
    assert plan.techniques == {"kivi": "plugin"}
    assert plan.engine["attention_backend"] == "CUSTOM"
    assert plan.engine["enable_prefix_caching"] is False
    assert plan.engine["block_size"] == 32  # lcm(16, group_size 32)
    assert plan.engine["additional_config"][CONFIG_KEY] == {"bits": 2, "group_size": 32, "residual_length": 128,
                                                            "layers": [], "num_layers": 6, "fused": False}
    assert "kv_cache_dtype" not in plan.engine
    assert any("saves no" in note or "fewer tokens" in note for note in plan.notes["kivi"])


def test_plan_serves_kivi_fused_with_the_packed_kernel(head32_dir: Path, tmp_path: Path) -> None:
    config = FusedKIVIConfig(bits=4, group_size=16, residual_length=32, layers=(2, 1))
    plan = to_vllm(economize.apply(_fused_base(head32_dir), config), tmp_path / "out")
    assert plan.techniques == {"kivi_fused": "plugin"}
    assert "block_size" not in plan.engine  # 16-token pages hold whole groups of 16
    entry = plan.engine["additional_config"][CONFIG_KEY]
    assert entry["fused"] is True and entry["layers"] == [1, 2] and entry["bits"] == 4
    assert any("paged KIVI kernel" in note for note in plan.notes["kivi_fused"])


def test_kivi_with_eagle3(head32_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(head32_dir), KIVIConfig(), Eagle3Config())
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.engine["attention_backend"] == "CUSTOM"
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("no rollback" in note for note in plan.notes["kivi"])


def test_kivi_and_streaming_are_refused_together(head32_dir: Path) -> None:
    model = economize.apply(_base(head32_dir), KIVIConfig(), KVEvictConfig(recent=32))
    with pytest.raises(UnsupportedBackendError, match="one attention backend per engine"):
        check_servable(model)


def test_small_heads_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "head16"
    make_tiny_model(seed=0).save_pretrained(folder)
    with pytest.raises(UnsupportedBackendError, match=r"head size 16.*at least 32"):
        check_servable(economize.apply(_base(folder), KIVIConfig()))


def test_sliding_layers_are_refused(tmp_path: Path) -> None:
    folder = _tiny_head32(tmp_path / "sliding", sliding=True)
    with pytest.raises(UnsupportedBackendError, match=r"layer 2 is a 'sliding' layer"):
        check_servable(economize.apply(_base(folder), KIVIConfig()))
    check_servable(economize.apply(_base(folder), KIVIConfig(layers=(1, 3, 5))))


def test_odd_group_sizes_are_refused(head32_dir: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match="power-of-two group_size"):
        check_servable(economize.apply(_base(head32_dir), KIVIConfig(group_size=24, residual_length=48)))


def test_support_table_and_plugin_step() -> None:
    rows = {row[0]: row for row in support_table(["kivi", "kivi_fused"])}
    assert rows["kivi"][1:3] == ("kv_cache", "plugin") and rows["kivi_fused"][1:3] == ("kv_cache", "plugin")
    assert "kivi extension" in [name for name, _ in plugin._steps()]


# ---- backend hooks ----------------------------------------------------------------------------------------------


def test_install_registers_the_custom_backend() -> None:
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum

    from economize.backends.vllm.kivi import install

    previous = _ATTN_OVERRIDES.get(AttentionBackendEnum.CUSTOM)
    try:
        install()
        assert _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM] == custom_attention.DISPATCH_PATH
        selected = custom_attention.resolve({CONFIG_KEY: KIVISettings(2, 32, 128).as_config()})
        assert selected is not None and selected.path == BACKEND_PATH
        assert resolve_obj_by_qualname(selected.path) is EconomizeKIVIBackend
        assert EconomizeKIVIBackend.get_name() == "CUSTOM"
    finally:
        if previous is None:
            _ATTN_OVERRIDES.pop(AttentionBackendEnum.CUSTOM, None)
        else:
            _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM] = previous


def test_spec_cells_are_widened() -> None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    settings = KIVISettings(2, 32, 128)
    SETTINGS.bind({CONFIG_KEY: settings.as_config()})
    spec = FullAttentionSpec(block_size=32, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    wide = EconomizeKIVIBackend.customize_spec(spec)
    size = cell_layout(64, torch.bfloat16, settings).size
    assert wide.state_content_bytes == size and wide.page_size_bytes == 2 * 32 * size
    assert EconomizeKIVIBackend.customize_spec(wide) == wide
    assert EconomizeKIVIBackend.supports_block_size(32) and not EconomizeKIVIBackend.supports_block_size(16)
    (multiple,) = EconomizeKIVIBackend.get_supported_kernel_block_sizes()
    assert multiple.base == 32
    with pytest.raises(ConfigError, match="equal key and value head sizes"):
        EconomizeKIVIBackend.customize_spec(
            FullAttentionSpec(block_size=32, num_kv_heads=2, head_size=64, head_size_v=32, dtype=torch.bfloat16))


def test_layer_names_map_to_covered_layers() -> None:
    assert layer_index("model.layers.3.self_attn.attn") == 3
    assert layer_index("layers.0.self_attn.attn") == 0 and layer_index("lm_head") is None
    assert layer_index("model.mylayers.3.attn") is None


def _builder(settings: KIVISettings | None, covers: bool, graphs: bool) -> EconomizeKIVIMetadataBuilder:
    builder = object.__new__(EconomizeKIVIMetadataBuilder)
    builder.kivi = settings
    builder.kivi_covers = covers
    builder.kivi_keys = 4096
    builder.kivi_prefill = torch.full((8,), 7, dtype=torch.int32)
    builder.decode_cudagraph_enabled = graphs
    return builder


def test_builder_marks_prefill_rows_and_draft_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    monkeypatch.setattr(TritonAttentionMetadataBuilder, "build", lambda self, prefix, common, fast_build=False:
                        SimpleNamespace(seq_lens=torch.zeros(3, dtype=torch.int32), max_seq_len=40))
    settings = KIVISettings(2, 32, 128)
    common = SimpleNamespace(is_prefilling=torch.tensor([True, False]))
    meta = _builder(settings, True, graphs=False).build(0, common)
    assert meta.kivi_quantize and meta.kivi_any_prefill and meta.kivi_keys == 40
    assert meta.kivi_prefill.tolist() == [1, 0, 0]
    meta = _builder(settings, True, graphs=True).build(0, SimpleNamespace(is_prefilling=torch.tensor([False] * 3)))
    assert not meta.kivi_any_prefill and meta.kivi_keys == 4096  # a CUDA graph is planned for max_model_len
    meta = _builder(settings, True, graphs=False).build(0, SimpleNamespace(is_prefilling=None))
    assert meta.kivi_prefill.tolist() == [1, 1, 1]  # without flags nothing is quantized
    drafter = _builder(settings, True, graphs=False)
    custom_attention.mark_draft_builders([[SimpleNamespace(metadata_builders=[drafter])]])
    assert not drafter.build(0, common).kivi_quantize
    assert not _builder(settings, False, graphs=False).build(0, common).kivi_quantize


def test_packed_kernel_runs_on_decoding_batches_only() -> None:
    impl = object.__new__(EconomizeKIVIImpl)
    impl.kivi = KIVISettings(2, 32, 128, fused=True)
    impl.alibi_slopes, impl.sliding_window, impl.chunk_lookback = None, (-1, -1), None
    meta = SimpleNamespace(kivi_any_prefill=False, max_query_len=4, mm_prefix_range_tensor=None, rswa_window=None,
                           causal=True)
    assert impl._packed_kernel_fits(meta, None)
    assert not impl._packed_kernel_fits(meta, torch.ones(1))  # quantized outputs stay with Triton
    for change in ({"kivi_any_prefill": True}, {"max_query_len": 65}, {"causal": False}):
        assert not impl._packed_kernel_fits(SimpleNamespace(**{**vars(meta), **change}), None)
    impl.kivi = KIVISettings(2, 32, 128)
    assert not impl._packed_kernel_fits(meta, None)  # kivi attends over the dense entries
