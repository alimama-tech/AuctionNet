import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_cql import run_cql

seed = 3
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    run_cql(seed=seed)