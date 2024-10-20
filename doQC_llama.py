
from QC_L1_svd import QC_L1
import torch
import numpy as np
import os
fpath = "/scratch/bcjw/bhuang4/cache/hub/models--meta-llama--Llama-2-7b/snapshots/backup/consolidated.00.pth"

rank = 10
model_data = torch.load(fpath,weights_only=True)
quant_list = ['wq','wk','wv','wo','w1','w2','w3']
for key, value in model_data.items():
    print("begin load")
    if any(substring in key for substring in quant_list):
        print(f"do {key}")
        # U,V,_,_,Wq = QC_L1(value,rank)
        QC_L1(value,rank,QC_max_iter=50)
        # print(value.shape)
        # value.shape()
        break
    # U,V,_,_,Wq = QC_L1(W,rank)
    # W_hat  = Wq+ U@V