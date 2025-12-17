import torch
from torch import Tensor

def noisybfactor(x: Tensor, b: Tensor, sigma: Tensor, simulator) -> Tensor:
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * simulator.scale    
        return x + error_new , sigma_new

