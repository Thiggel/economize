```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize

device = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
ids = tokenizer("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                "How many clips did Natalia sell altogether?", return_tensors="pt").input_ids.to(device)
```
