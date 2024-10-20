import torch
import time


def shrink(weight, threshold):
    return torch.sign(weight) * torch.nn.functional.relu(torch.abs(weight) - threshold)

def quantize(Win, z, s):
    return torch.round(Win * s) + z

def de_quantize(Win, z, s):
    return (Win - z) / s

def QC_L1(W,r, QC_max_iter = 5,device='cuda'):
    W = W.to(device)  # Ensure W is on GPU if available
    nbits = 3
    max_v = round(2 ** nbits - 1)
    
    # _min and _max computation
    _min = torch.min(W, dim=1, keepdim=True).values
    _max = torch.max(W, dim=1, keepdim=True).values
    s = torch.clip(max_v / (_max - _min), max=2e4)
    z = -_min * s
    
    beta = 1e-1
    kappa = 0.99
    E = torch.zeros_like(W, device=device)  # E initialized to zeros
    L1norm_my = []
    
    bt = time.time()
    
    for QC_iter in range(QC_max_iter):
        # Quantization loop
        for Q_iter in range(20):
            Wq = quantize(W - E, z, s)
            Wdq = de_quantize(Wq, z, s)
            
            # Update We
            We = shrink(W - E - Wdq, 1.0/beta)
            beta = beta * kappa
            
            # Update z
            z = torch.mean(Wq - (W - E - We) * s, dim=1, keepdim=True)
        
        if QC_iter == 0:
            tmp = de_quantize(quantize(W, z, s), z, s)
            print(f"Initial L1 norm: {torch.norm(W - tmp, p=1).item()}")
        # Decomposition
        Wdq_new = de_quantize(quantize(W - E, z, s), z, s)
        U, S, V = torch.linalg.svd(W.float() - Wdq_new.float(), full_matrices=False)  #do SVD to error matrix
        S = torch.diag(S)
        U_h = U[:,:r] @ torch.sqrt(S[:r,:r]) # calculate the U_h and V_h according to rank
        V_h = torch.sqrt(S[:r,:r]) @ V[:r,:]
        E = U_h @ V_h

        L1norm_my.append(torch.norm(W - E - Wdq_new, p=1).item())
        print(f"Iter {QC_iter:02d} L1 norm: {L1norm_my[-1]:.3f}")

    et = time.time()
    print(f"Took {et - bt:.3f}s")

    # return result.A,result.B.T,z,s,Wdq_new

