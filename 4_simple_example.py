import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition

info = np.array(["S1", "S2", "S3", "S4", "S5", "S6"])

data_set_1 = np.array([[150, 155, 153, 179, 176, 170],
                       [51, 56, 52, 77, 75, 68],
                       [30, 34, 35, 60, 65, 65],
                       [40, 48, 65, 71, 68, 30]]).transpose()

mean = np.nanmean(data_set_1, axis=0)
std = np.nanstd(data_set_1, axis=0)

data_set_1_std = (data_set_1 - mean) / std

model = decomposition.PCA(n_components=2)
model.fit(data_set_1_std)
z = model.transform(data_set_1_std)
print(model.components_)

plt.figure()
for i in range(z.shape[0]):
    plt.scatter(z[i, 0], z[i, 1], label=info[i])
    plt.text(z[i, 0], z[i, 1], s=info[i])
plt.legend()

