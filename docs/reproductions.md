# Reproductions

This page compares numbers published in the papers behind economize's techniques with the same quantities measured
with economize on the same model. Each study names the model, the hardware, the metric, the command that produced
the measurement and a result file with every configuration and every metric of the run. The tables are generated
when the site is built (`docs/hooks/reproductions.py`) from the data file `docs/data/reproductions.yaml`, which
contributors extend by pull request.

A study lists only the settings for which the paper's number appears in this documentation, and every economize number
is a measurement. Results over several calibration seeds give the mean and the sample standard deviation. Where the
unmodified model already scores differently from the paper (for example because the evaluation text differs), the
change from the unmodified model is the quantity to compare, and the tables show it next to the absolute values.

The studies below were measured with the evaluation harness of the economize authors. The harness is not yet
public, so each study's command names the harness's entry point. The result files list the full configuration of every
variant, which together with the configuration lines of each study is enough to rerun a study with the public API (`economize.apply` or `economize.load`, see [Weights and activations](weights.md) and [Drafters](drafters.md)).
economize was released under the name effkit before version 0.3.0, so the result files of version 0.2.0 were produced
by effkit 0.2.0; their copies here rename that key. The quantization studies use the reference numerics of each format
(`backend="fake"`), which reproduce the quantized model's outputs without its memory saving, so they measure quality
only. The drafting studies measure wall-clock speed, and [Benchmarks](benchmarks.md) explains why the speed of a
drafter depends on the stack around it.

<!-- reproductions-table -->

## Adding a reproduction

A reproduction is one entry under `studies:` in `docs/data/reproductions.yaml` plus its result file. The steps:

1. Run the technique on a model for which the paper reports a number, with the paper's evaluation setup as far as it
   is known. Any script works; [Examples](examples.md) has a perplexity script (`examples/compress.py`) and a speed
   benchmark (`examples/benchmark.py`) to start from.
2. Save the run's output as a JSON file under `docs/data/results/` (or publish it elsewhere and use its URL). It should
   hold the versions of economize, torch, transformers and CUDA, the GPU, the date, the configuration of every variant
   and every metric, and no machine paths.
3. Add a study to `docs/data/reproductions.yaml` with the fields below, and quote the paper's number together with
   its table or figure in the technique's tutorial or in [Weights and activations](weights.md) if the documentation
   does not quote it yet, so a reader can check the pairing.
4. Run `mkdocs build --strict`. The hook validates the data file and stops the build with a message that names the
   study and the field if an entry does not follow the schema. `pytest tests/docs/test_reproductions.py` runs the
   same checks.

A study has these fields (schema version 1, the top-level key `schema: 1`):

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique identifier, used as the anchor of the study's section |
| `title` | yes | Heading of the study |
| `technique` | yes | Name of the technique in `economize.TECHNIQUES` (the build checks it) |
| `model` | yes | Hub id of the base model and its dtype |
| `hardware` | yes | GPU (or CPU) the measurement ran on |
| `metric` | yes | Metric and evaluation setup |
| `better` | yes | `lower` or `higher` |
| `paper` | yes | `source` (paper and table), and optionally `arxiv` (an arXiv id) or `url`, and `baseline` (the paper's number for the unmodified model) |
| `economize` | yes | `version`, `date` of the measurement, and optionally `baseline` (the measured number for the unmodified model) |
| `command` | yes | The command that produced the result |
| `result` | yes | Path under `docs/` of the result file, or a URL |
| `rows` | yes | One entry per setting: `setting`, `config` (the economize configuration), `paper` and `economize` |
| `note` | no | Differences in setup that a reader needs to interpret the comparison |
| `digits` | no | Decimal places of the rendered numbers (default 2) |
| `unit` | no | Suffix of the rendered numbers, such as `x` for speedups |

A value under `paper`, `economize` or `baseline` is a number, or for a result over several seeds a mapping
`{mean: ..., std: ..., seeds: ...}` with the sample standard deviation and at least two seeds. Give the change from
the unmodified model by setting both baselines; the tables then add the change columns.

```yaml
schema: 1
studies:
  - id: wanda-llama-7b
    title: Unstructured and 2:4 pruning of LLaMA-7B
    technique: prune
    model: huggyllama/llama-7b (fp16)
    hardware: NVIDIA A100-SXM4-40GB
    metric: WikiText-2 test perplexity, 2048-token windows
    better: lower
    paper: {source: "Wanda, Table 3", arxiv: "2306.11695", baseline: 5.68}
    economize: {version: 0.3.1, date: 2026-10-12, baseline: 5.677}
    command: python my_eval.py --method wanda --sparsity 0.5
    result: data/results/my_run.json
    rows:
      - setting: Wanda, 50% unstructured
        config: PruneConfig(method="wanda", sparsity=0.5)
        paper: 7.26
        economize: {mean: 7.261, std: 0.014, seeds: 3}
```
