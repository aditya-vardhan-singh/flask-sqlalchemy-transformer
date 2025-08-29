# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, torch_dtype="auto"
)