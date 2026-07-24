import torch, subprocess, sys
print("HELLO_FROM_POD", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
