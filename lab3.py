#3rd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data =load_iris()

X = PCA(2).fit_transform(data.data)

for i, c in zip(range (3), 'rgb'):

  plt.scatter (*X[data.target == i].T, c=c, label=data.target_names[i])

plt.title('Iris PCA'),

plt.xlabel('PC1'), plt.ylabel('PC2')

plt.legend(), plt.grid(True)

plt.show()