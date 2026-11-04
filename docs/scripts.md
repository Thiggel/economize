# Scripts

Complete scripts from the [`examples/`](https://github.com/Thiggel/economize/tree/main/examples) directory of the
repository. Each takes `--help` and runs on a GPU when one is visible, on the CPU otherwise. The tests run all of
them on a CPU with tiny settings (`tests/examples/test_scripts.py`).

```bash
python examples/prepare_data.py --out data
python examples/train_drafter.py --technique eagle3 --data data --out runs/eagle3
python examples/generate.py --drafter runs/eagle3/final
```

## Tokenize a dataset

`examples/prepare_data.py`

```python
--8<-- "examples/prepare_data.py"
```

## Generate with a drafter

`examples/generate.py`

```python
--8<-- "examples/generate.py"
```

## Lossless sampling and batches

`examples/sampling.py`

```python
--8<-- "examples/sampling.py"
```

## Train a drafter

`examples/train_drafter.py`

```python
--8<-- "examples/train_drafter.py"
```

## Compress the weights

`examples/compress.py`

```python
--8<-- "examples/compress.py"
```

## Combine techniques

`examples/combine.py`

```python
--8<-- "examples/combine.py"
```

## Benchmark

`examples/benchmark.py`

```python
--8<-- "examples/benchmark.py"
```

## A new training objective

`examples/custom_objective.py`

```python
--8<-- "examples/custom_objective.py"
```

## Serve with vLLM

`examples/serve_vllm.py`

```python
--8<-- "examples/serve_vllm.py"
```
