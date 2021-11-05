import random
import torch
import numpy as np
from matplotlib import pyplot as plt
import umap
import hdbscan
from sklearn.cluster import KMeans


fingerpaths = []
with open("filtered_paths.txt", "r") as f:
    fingerpaths = [line.split(",")[0] for line in f.readlines()]
print("num of files:", len(fingerpaths))

random.shuffle(fingerpaths)
sample_of_paths = fingerpaths[:100000]
#sample_of_timeseries = []
sample_of_gt = []
for p in sample_of_paths:
    whole = torch.from_numpy(np.load(p))
    #print("whole shape", whole.shape) # torch.Size([1003])
    x = whole[:1000]
    #print("x shape", x.shape) # torch.Size([1000])
    gt = whole[1000:-1]
    #print("gt shape", gt.shape) # torch.Size([2])
    #sample_of_timeseries.append(x)
    sample_of_gt.append(gt)
#print(len(sample_of_timeseries))
#eg_x = sample_of_timeseries[0]
#print("x shape", eg_x.shape)
eg_gt = sample_of_gt[0]
print("gt shape", eg_gt.shape)

#whole_series_tensor = torch.stack(sample_of_timeseries)

kmeans = KMeans(n_clusters=10).fit(torch.stack(sample_of_gt))
print("kmeans labels", kmeans.labels_[:5])
predictions = kmeans.predict(torch.stack(sample_of_gt[:5]))
print("predictions:", predictions)
print(fingerpaths[0])
print(sample_of_gt[0])

rest_of_gt = []
for k in range(100000, len(fingerpaths)): 
    whole = torch.from_numpy(np.load(fingerpaths[k]))
    gt = whole[1000:-1]
    rest_of_gt.append(gt)

print("rest_of_gt len", len(rest_of_gt))
rest_of_paths = fingerpaths[100000:]
print("rest_of_paths len", len(rest_of_paths))
rest_of_predictions = kmeans.predict(torch.stack(rest_of_gt))

with open ("clustered_paths.txt", "w") as f:
    for i in range(len(sample_of_paths)):
        line = f"{sample_of_paths[i]},{int(kmeans.labels_[i])}"
        f.write(line + "\n")

    for j in range(len(rest_of_gt)):
        line = f"{rest_of_paths[j]},{int(rest_of_predictions[j])}"
        f.write(line + "\n")




# whole_gt_tensor = torch.stack(sample_of_gt)
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(
#     whole_gt_tensor[:, 0],
#     whole_gt_tensor[:, 1]
# )
# plt.savefig("gt_scatter.png")

# reducer_big = umap.UMAP(n_components=64)
# embedding_big = reducer_big.fit_transform(whole_series_tensor)
# print("embedding_big shape", embedding_big.shape)

# # fig, ax = plt.subplots(figsize=(10,10))
# # ax.scatter(
# #     embedding[:, 0],
# #     embedding[:,1],
# #     s = 2
# # )
# # plt.savefig("umap_scatter.png")

# clusterer = hdbscan.HDBSCAN()
# clusterer.fit(embedding_big)
# cluster_labels = clusterer.labels_


# # do small embedding here
# reducer_small = umap.UMAP()
# embedding_small = reducer_small.fit_transform(whole_series_tensor)

# fig, ax = plt.subplots(figsize=(10,10))
# for label in np.unique(cluster_labels):
#     idx = np.argwhere(cluster_labels == label)
#     ax.scatter(
#         embedding_small[idx, 0],
#         embedding_small[idx, 1],
#         label=label
#     )
# plt.savefig("labelled_cluster.png")


