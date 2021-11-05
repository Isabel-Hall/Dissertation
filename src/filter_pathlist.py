import numpy as np

with open("paths.txt", "r") as f:
    lines = [line.strip("\n") for line in f.readlines()]

    #lines = lines[:20]

#print(lines[:10])

# 1s are empty fingerprints
zero_count = 0
with open("filtered_paths.txt", "w") as f:
        
    for line in lines:
        #print(line.split(",")[1])
        if int(line.split(",")[1]) == 0:
            f.write(line + "\n")
        elif zero_count < 230000 and int(line.split(",")[1]) == 1:
            f.write(line + "\n")
            zero_count += 1


    print(zero_count)