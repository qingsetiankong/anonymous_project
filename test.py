import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.") 
else:
    print("CUDA is not available. CPU will be used.")
path = r"/media/ubuntu20/D/robotic/复现/复现/target_pose_bank/target_pose_bank.npy"
x = np.load(path)
print(x)