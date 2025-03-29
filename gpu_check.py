import torch
print(torch.cuda.is_available())  # Should be False on Mac if using non-NVIDIA GPU
print(torch.backends.mps.is_available())  # True if MPS (Apple GPU) is available on Apple Silicon

