import torch

print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.") 
else:
    print("CUDA is not available. CPU will be used.")