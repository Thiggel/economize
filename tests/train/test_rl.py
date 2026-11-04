"""On-policy RL pieces (:mod:`economize.train.rl`) and the ``jump_aware`` objective."""

from __future__ import annotations

import pytest
import torch

import economize
from economize.errors import IncompatibleTechniquesError
from economize.runtime.cache import CacheSpec, HybridCache
from economize.techniques.jump import JumpAware, JumpRollout, hybrid_log_probs, jump_acceptance
from economize.train import OBJECTIVES, Batch, TeacherRequest, policy_pass, run_teacher
from economize.train.rl import (
    acceptance,
    cycle_starts,
    detach_outputs,
    detached_cache,
    expected_tokens,
    token_log_probs,
)
from helpers import PROMPT_LEN, VOCAB, make_identity_jump, make_tiny_model, perturb


def _jump_model(seed: int = 0, m: int = 2, u: int = 4, adaptation: str = "frozen") -> economize.EfficientModel:
    base = make_tiny_model(seed=seed)
    config = economize.JumpConfig(target_layer=m, source_layer=u, adaptation=adaptation, lora_rank=2)
    model = economize.apply(base, config)
    perturb(model.techniques["jump"].predictor, std=0.2, seed=seed + 1)
    return model


def _request(model: economize.EfficientModel) -> TeacherRequest:
    jump = model.techniques["jump"]
    return TeacherRequest(hidden_layers=tuple(sorted({jump.target_layer, *jump.source_layers})), cache=True)


def _ids(rows: int = 2, length: int = 40, seed: int = 3) -> torch.Tensor:
    return torch.randint(0, VOCAB, (rows, length), generator=torch.Generator().manual_seed(seed))


def test_policy_pass_matches_teacher_and_has_gradients() -> None:
    model = _jump_model()
    model.base_model.requires_grad_(True)
    batch = Batch(_ids())
    request = _request(model)
    reference = run_teacher(model, batch, request)
    outputs = policy_pass(model, batch, request)
    assert outputs.final.requires_grad and not reference.final.requires_grad
    torch.testing.assert_close(outputs.final, reference.final)
    for layer in request.hidden_layers:
        torch.testing.assert_close(outputs.hidden[layer], reference.hidden[layer])
    log_prob, entropy = token_log_probs(outputs, batch.input_ids)
    hf = model.base_model(batch.input_ids).logits.float().log_softmax(-1)[:, :-1]
    torch.testing.assert_close(log_prob, hf.gather(-1, batch.input_ids[:, 1:, None])[..., 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy, -(hf.exp() * hf).sum(-1), atol=1e-5, rtol=1e-5)
    log_prob.sum().backward()
    assert model.adapter.layer(1).self_attn.q_proj.weight.grad is not None


def test_token_log_probs_chunks_agree() -> None:
    model = _jump_model()
    model.base_model.requires_grad_(True)
    batch = Batch(_ids())
    whole = token_log_probs(policy_pass(model, batch, _request(model)), batch.input_ids, chunk=0)
    whole[0].sum().backward()
    grad = model.adapter.lm_head.weight.grad.clone()
    model.base_model.zero_grad()
    parts = token_log_probs(policy_pass(model, batch, _request(model)), batch.input_ids, chunk=10)
    parts[0].sum().backward()
    torch.testing.assert_close(parts[0], whole[0])
    torch.testing.assert_close(parts[1], whole[1])
    torch.testing.assert_close(model.adapter.lm_head.weight.grad, grad, atol=1e-5, rtol=1e-4)


def test_acceptance_and_expected_tokens() -> None:
    p = torch.tensor([0.5, 0.5, 0.0]).clamp_min(1e-30).log()
    q = torch.tensor([0.25, 0.25, 0.5]).log()
    assert acceptance(p, q).item() == pytest.approx(0.5)
    accept = torch.tensor([1.0, 0.5, 0.5])
    torch.testing.assert_close(expected_tokens(accept), torch.tensor([2.0, 2.5, 2.75]))


def test_identity_drafter_is_always_accepted() -> None:
    model = make_identity_jump(make_tiny_model())
    ids = _ids()
    outputs = run_teacher(model, Batch(ids), _request(model))
    starts = torch.arange(5, 30, 4)
    accept = jump_acceptance(model, model.techniques["jump"], ids, outputs, starts, 6)
    assert accept.shape == (6, 2, len(starts))
    torch.testing.assert_close(accept, torch.ones_like(accept), atol=1e-4, rtol=0)
    tau = expected_tokens(accept.mean((1, 2)))
    torch.testing.assert_close(tau, torch.arange(2, 8, dtype=torch.float32), atol=1e-3, rtol=0)


def test_hybrid_with_cycle_one_is_exact() -> None:
    model = _jump_model()
    ids = _ids()
    outputs = run_teacher(model, Batch(ids), _request(model))
    exact = token_log_probs(outputs, ids)
    hybrid = hybrid_log_probs(model, model.techniques["jump"], ids, outputs, PROMPT_LEN, 1)
    torch.testing.assert_close(hybrid[0], exact[0])


class _Recorder(economize.TopPSampler):
    """A sampler that records the log-probabilities of every distribution it samples from, in order."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_probs: list[torch.Tensor] = []

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        self.log_probs.append((logits.float() / self.temperature).log_softmax(-1).view(-1))
        return super().__call__(logits, history)

    def select_many(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[0] == 1
        self.log_probs.append((logits.float() / self.temperature).log_softmax(-1)[0])
        return super().select_many(logits)


@pytest.mark.parametrize("cycle", [2, 3, 4])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_hybrid_log_probs_match_lossy_decoding(cycle: int, temperature: float) -> None:
    """The distributions that lossy cycle decoding samples from, recorded during decoding, equal the hybrid
    log-probabilities computed afterwards from one teacher-forced pass and one rollout."""
    model = _jump_model(seed=5)
    jump = model.techniques["jump"]
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(7))
    recorder = _Recorder(temperature=temperature, seed=11)
    config = economize.DecodeConfig(cycle_length=cycle, lossless=False)
    result = model.decode(prompt, 21, decoding=config, selector=recorder)
    tokens = result.tokens.view(-1)
    assert len(recorder.log_probs) == len(tokens) == 21
    ids = torch.cat([prompt, tokens, torch.zeros(cycle, dtype=torch.long)])[None]
    outputs = run_teacher(model, Batch(ids), _request(model))
    log_prob, entropy = hybrid_log_probs(model, jump, ids, outputs, PROMPT_LEN, cycle, temperature=temperature)
    recorded = torch.stack(recorder.log_probs)
    span = slice(PROMPT_LEN - 1, PROMPT_LEN - 1 + len(tokens))
    torch.testing.assert_close(log_prob[0, span], recorded.gather(-1, tokens[:, None])[:, 0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(entropy[0, span], -(recorded.exp() * recorded).sum(-1), atol=1e-4, rtol=1e-4)
    # the drafts differ from the exact policy (the test would pass trivially otherwise)
    exact = token_log_probs(outputs, ids, temperature=temperature)[0]
    drafted = [t for t in range(span.start, span.stop) if (t - span.start) % cycle]
    assert (exact[0, drafted] - log_prob[0, drafted]).abs().max() > 1e-2
    assert len(cycle_starts(PROMPT_LEN, ids.shape[1], cycle)) * cycle >= len(tokens)


def test_hybrid_log_probs_train_the_whole_hybrid_policy() -> None:
    model = _jump_model()
    model.base_model.requires_grad_(True)
    jump = model.techniques["jump"]
    ids = _ids()
    outputs = policy_pass(model, Batch(ids), _request(model))
    log_prob, _ = hybrid_log_probs(model, jump, ids, outputs, PROMPT_LEN, 4)
    starts = cycle_starts(PROMPT_LEN, ids.shape[1], 4)
    log_prob[:, starts + 1].sum().backward()  # drafted tokens only (the first draft step of every cycle)
    assert all(p.grad is not None for p in jump.predictor.parameters())
    assert model.adapter.layer(1).mlp.up_proj.weight.grad is not None  # below m, through the source states
    assert model.adapter.layer(5).mlp.up_proj.weight.grad is not None  # on the jump path


def test_detached_teacher_trains_only_the_drafter() -> None:
    """Variant c of the RL pilot: the drafter loss on a detached policy pass leaves the policy's gradients alone."""
    model = _jump_model()
    model.base_model.requires_grad_(True)
    jump = model.techniques["jump"]
    ids = _ids()
    outputs = policy_pass(model, Batch(ids), _request(model))
    teacher = detach_outputs(outputs)
    assert not teacher.final.requires_grad and not any(h.requires_grad for h in teacher.hidden.values())
    assert not any(k.requires_grad for k in teacher.cache.keys.values())
    torch.testing.assert_close(teacher.final, outputs.final)
    objective = JumpRollout(horizon=3, stride=4)
    drafter = [p for p in jump.parameters() if p.requires_grad]
    out = objective.loss(model, Batch(ids), teacher)
    out.loss.backward(inputs=drafter)
    assert all(p.grad is not None for p in drafter)
    assert all(p.grad is None for p in model.base_model.parameters())


def test_jump_aware_trains_the_shared_layers() -> None:
    assert "jump_aware" in OBJECTIVES.names()
    model = _jump_model(m=2, u=4)
    objective = JumpAware(horizon=3, stride=4)
    objective.setup(model)
    groups = {g.name: g for g in objective.parameter_groups(model)}
    assert set(groups) == {"drafter", "shared"} and groups["shared"].kind == "backbone"
    shared = {id(p) for p in groups["shared"].params}
    assert id(model.adapter.layer(3).self_attn.q_proj.weight) in shared
    assert id(model.adapter.layer(2).self_attn.q_proj.weight) not in shared
    assert id(model.adapter.lm_head.weight) in shared
    for group in groups.values():
        for p in group.params:
            p.requires_grad_(True)
    ids = _ids()
    out = objective.loss(model, Batch(ids), run_teacher(model, Batch(ids), objective.teacher_request(model)))
    out.loss.backward()
    assert model.adapter.layer(3).mlp.up_proj.weight.grad is not None
    assert model.adapter.layer(1).mlp.up_proj.weight.grad is None  # the teacher is under stop-gradient
    with pytest.raises(IncompatibleTechniquesError):
        JumpAware().setup(_jump_model(adaptation="lora"))


def test_cycle_starts_of_a_short_generation_are_empty() -> None:
    """A generation shorter than one cycle has no cycle start (``torch.arange`` raised before)."""
    assert len(cycle_starts(10, 11, 4)) == 0
    assert cycle_starts(4, 20, 4).tolist() == [3, 7, 11, 15]


# ---- detached teacher caches -----------------------------------------------------------------------------------------
def test_detached_cache_keeps_the_recurrent_states_of_a_hybrid_cache() -> None:
    """The RL teacher of a hybrid model (gated delta nets, Mamba-2) detaches a :class:`HybridCache`: same kind, same
    buffers and recurrent states, no autograd history."""
    cache = HybridCache(CacheSpec(8, 2, "hybrid"))
    weight = torch.ones((), requires_grad=True)
    cache.keys[0], cache.values[0] = torch.randn(2, 1, 9, 4) * weight, torch.randn(2, 1, 9, 4) * weight
    cache.conv_states[1], cache.recurrent_states[1] = torch.randn(2, 6, 4) * weight, torch.randn(2, 2, 4, 4) * weight
    copy = detached_cache(cache)
    assert type(copy) is HybridCache and copy.spec == cache.spec
    for name in ("keys", "values", "conv_states", "recurrent_states"):
        original, detached = getattr(cache, name), getattr(copy, name)
        assert detached.keys() == original.keys()
        for layer, tensor in detached.items():
            assert not tensor.requires_grad and torch.equal(tensor, original[layer])
