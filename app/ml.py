# # Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained(
#     "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, torch_dtype="auto"
# )

import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    torch_dtype=torch.float32  # or torch.float16 if you want FP16
)
