import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

counter = 0
for p in Path("/pytorch_data/train/").iterdir():
    if p.is_file():
        print(p)
        x = torch.load(p).view(1003, -1)
        #print("x shape", x.shape)
        fingerprints = torch.unbind(x, dim=1)
        #print("fingerprints", len(fingerprints))
        #print(fingerprints[0].shape)
        for f in fingerprints:
            #print("f", f.shape)
            #input("...")
            np.save(f"/pytorch_data/individual/train/{counter}.pt", f)
            #torch.save(f, f"/pytorch_data/individual/train/{counter}.pt")
            counter += 1


