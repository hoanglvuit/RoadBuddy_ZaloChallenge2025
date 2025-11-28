import torch
import numpy as np
import random
import os
import math

def set_seed(seed: int = 42):
    # Python built-in RNG
    random.seed(seed)
    
    # Numpy RNG
    np.random.seed(seed)
    
    # PyTorch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # For deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For reproducibility across hash-based operations (Python ≥3.9)
    os.environ["PYTHONHASHSEED"] = str(seed)
