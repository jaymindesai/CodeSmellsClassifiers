import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import sqrt
from imblearn.over_sampling import SMOTE

from skfeature.function.statistical_based import CFS
from skfeature.function.wrapper import decision_tree_forward as dtf
from skfeature.function.wrapper import svm_forward as svmf

# df = pd.read_csv('../_data/cl-data-class.csv')
# df = pd.read_csv('../data/CSV/FeatureEnvy/_feature-envy.csv')
# df = pd.read_csv('../data/CSV/GodClass/_god-class.csv')
# df = pd.read_csv('../data/CSV/LongMethod/_long-method.csv')

# to_drop_data_god = ['$isStatic_type']
# to_drop_feature_long = ['$isStatic_type', '$isStatic_method']

# df.drop(columns=to_drop_data_god, inplace=True)
# df.drop(columns=to_drop_feature_long, inplace=True)

# Combined Data
df = pd.read_csv('../_data/cl-data-class.csv')
# df = pd.read_csv('../_data/cl-god-class.csv')
# df = pd.read_csv('../_data/ml-feature-envy.csv')
# df = pd.read_csv('../_data/ml-long-method.csv')

y = df['SMELLS']
X = df.drop(columns=['SMELLS'])

rows, cols = X.shape
num_feats = int(cols ** 0.5)

# X_smote, y_smote = SMOTE().fit_resample(X, y)

# cfs_feats = CFS.cfs(X.values, y.values)
# cfs_X = X.iloc[:, cfs_feats]
# X_smote, y_smote = SMOTE().fit_resample(cfs_X, y)

dtf_feats = dtf.decision_tree_forward(X.values, y.values, num_feats)
dtf_X = X.iloc[:, dtf_feats]
X_smote, y_smote = SMOTE().fit_resample(dtf_X, y)

# svmf_feats = svmf.svm_forward(X.values, y.values, num_feats)
# svmf_X = X.iloc[:, svmf_feats]
# X_smote, y_smote = SMOTE().fit_resample(svmf_X, y)

Sum_of_squared_distances = []

clusters = int(sqrt(len(X_smote)))

K = range(1, clusters)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_smote)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()