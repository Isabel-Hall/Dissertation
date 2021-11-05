from pathlib import Path
import numpy as np

lines = []
t1_avgs = []
t2_avgs = []
t1_stds = []
t2_stds = []

for p in Path("/media/DataSSD/datasets/MRI_Issie/numpy/").iterdir():
    f = np.load(p)
    #print(np.shape(f))
    t1_avg = np.average(f[:,:,0])
    t1_avgs.append(t1_avg)
    t1_std = np.std(f[:,:,0])
    t1_stds.append(t1_std)
    #print(t1_avg)
    #print(t1_std)
    t2_avg = np.average(f[:,:,1])
    t2_avgs.append(t2_avg)
    t2_std = np.std(f[:,:,1])
    t2_stds.append(t2_std)
    #print(t2_avg)
    #print(t2_std)
    lines.append((p.stem, "t1_avg:", str(t1_avg), "t1_std:", str(t1_std), "t2_avg:", str(t2_avg), "t2_std:", str(t2_std)))


#print(t1_avgs)
overall_t1_avg = np.average(np.asarray(t1_avgs))
overall_t1_std = np.average(np.asarray(t1_stds))
overall_t2_avg = np.average(np.asarray(t2_avgs))
overall_t2_std = np.average(np.asarray(t2_stds))

with open("groundtruth_avgs.txt", "w") as g:
    for line in lines:
        g.write(" ".join(line) + "\n")
    g.write("overall_t1_avg:" + str(overall_t1_avg) + "\n")
    g.write("overall_t1_std:" + str(overall_t1_std) + "\n")
    g.write("overall_t2_avg:" + str(overall_t2_avg) + "\n")
    g.write("overall_t2_std:" + str(overall_t2_std) + "\n")

print("overall_t1_avg", overall_t1_avg)
print("overall_t1_std", overall_t1_std)
print("overall_t2_avg", overall_t2_avg)
print("overall_t2_std", overall_t2_std)

# overall_t1_avg 0.5113866338173979
# overall_t1_std 0.7184752571113431
# overall_t2_avg 0.06698109627281001
# overall_t2_std 0.12903516669471898