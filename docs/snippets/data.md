Training data is a flat file of `uint32` token ids, the format of nanoGPT-style pretraining shards, and calibration
data is a `[N, S]` tensor of token ids. Both are made here from GSM8K:

```python
import numpy as np
from datasets import load_dataset

texts = [f"{row['question']}\n{row['answer']}" for row in load_dataset("openai/gsm8k", "main", split="train")]
tokens = np.concatenate([tokenizer(t + tokenizer.eos_token).input_ids for t in texts]).astype(np.uint32)
tokens[: len(tokens) * 9 // 10].tofile("train.bin")
tokens[len(tokens) * 9 // 10 :].tofile("val.bin")

calibration = torch.from_numpy(tokens[: 128 * 2048].astype(np.int64)).view(128, 2048)
```
