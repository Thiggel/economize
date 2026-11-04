# Lookahead decoding

<!-- technique-info: lookahead -->

Lookahead decoding (Fu et al., 2024) drafts without a second network. Every cycle runs one exact pass of the model
over two branches that do not attend to each other. The lookahead branch holds a 2D Jacobi window of `window`
columns and `ngram - 1` levels of the model's own guesses for future tokens. The pass refines the window by one
Jacobi iteration, and every column then yields an n-gram of `ngram` tokens for an n-gram pool. The verification
branch holds the pool's n-grams that start with the last committed token, merged into one tree, and the verifier
keeps the longest path the model agrees with. Window tokens are never accepted themselves, so greedy decoding is
lossless. The method trades extra positions in each forward pass for fewer passes. It has no parameters and needs
no training or data.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

An n-gram proposes `ngram - 1` tokens, so `cycle_length=ngram` lets one cycle accept all of them:

```python
from economize.techniques.lookahead import LookaheadConfig

model = economize.apply(base, LookaheadConfig(window=7, ngram=5), decoding=economize.DecodeConfig(cycle_length=5))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # model calls and the positions they ran
```

The cost summary counts every node of the exact pass, the window included, since that is the compute the method
spends. The output is the model's greedy output token for token.

## Decoding options

The window size is the main setting. A larger `window` and `ngram` make more and longer guesses per pass but widen
every pass. The paper's settings on A100 GPUs were `window=15, ngram=5` for 7B models, `window=10` for 13B models
and `window=7` for 34B models, each with `guesses` equal to `window`. `guesses` caps the number of n-grams the pool
keeps per first token and the verification branch checks. `pool_from_prompt=True` also fills the pool with every
n-gram of the prompt and the output so far, as the official implementation's `POOL_FROM_PROMPT` option does, which
helps when the output repeats the input:

```python
model.detach()
model = economize.apply(base, LookaheadConfig(window=15, ngram=5, guesses=15, pool_from_prompt=True),
                        decoding=economize.DecodeConfig(cycle_length=5))
result = model.decode(ids[0], 64)
```

The drafts are deterministic, so the `rejection` verifier treats them as point masses and stays lossless at a
temperature above zero. Verifying a draft tree by rejection sampling needs a selector that gives the sampling
distribution of every node, such as `economize.TopPSampler`, so sampled runs go through `model.decode`:

```python
sampling = economize.DecodeConfig(cycle_length=5, verifier="rejection")
sampler = economize.TopPSampler(temperature=0.7, top_p=0.9, seed=0, device="cuda")
sampled = model.decode(ids[0], 64, decoding=sampling, selector=sampler)
```

Lookahead decoding needs a verifier that accepts trees (`greedy`, `rejection` or `typical`).

## Saving and loading

The technique has no weights, so a saved model holds only its configuration:

```python
model.save_pretrained("lookahead_model")
model.detach()

model = economize.load(base, "lookahead_model", decoding=economize.DecodeConfig(cycle_length=5))
```

## Combining with other techniques

Lookahead decoding is a drafter, so it does not combine with a second drafter. It composes with techniques that
change the model or its cache, such as weight quantization:

```python
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    LookaheadConfig(window=7, ngram=5),
    decoding=economize.DecodeConfig(cycle_length=5),
)
result = model.decode(ids[0], 64)
```

## Serving

vLLM has no Jacobi window. With `pool_from_prompt=True` the export serves the pool built from the committed tokens
through vLLM's n-gram proposer, which drops the lookahead branch. `economize.backend_support` reports what each
runtime runs:

```python
print(economize.backend_support(LookaheadConfig(pool_from_prompt=True)))
```

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

model.detach()
model = economize.apply(base, LookaheadConfig(window=7, ngram=5, pool_from_prompt=True))
plan = serving.to_vllm(model, "out/qwen-lookahead", num_speculative_tokens=4)
llm = plan.llm()
```

## Configuration

<!-- technique-config: lookahead -->

## Reference

Yichao Fu, Peter Bailis, Ion Stoica and Hao Zhang. *Break the Sequential Dependency of LLM Inference Using Lookahead
Decoding.* ICML 2024, arXiv:2402.02057.
