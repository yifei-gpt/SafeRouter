"""One file per trainable router, plus the cross-run pooling."""
import torch

CORRECT_THR = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
