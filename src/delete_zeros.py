import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


num_to_keep = 230_000
num_kept = 0


def check_if_zero(path_to_fingerprint):

    fingerprint = np.load(path_to_fingerprint)
    ground_truth = fingerprint[1000:-1]
    is_zero = np.all(ground_truth == 0)

    return path_to_fingerprint, is_zero


lines = []
with ThreadPoolExecutor() as executor:

    futures = []
    count = 0
    for path in Path("/media/DataSSD/datasets/MRI_Issie/individual/train/").iterdir():
        future = executor.submit(check_if_zero, path)
        futures.append(future)
        # count += 1
        # if count >= 10:
        #     break

    with open("./paths.txt", "w") as f:
        for future in futures:
            (p, is_zero) = future.result()
            #print(p, is_zero)
            line = f"{p},{int(is_zero)}"
            f.write(line + "\n")


