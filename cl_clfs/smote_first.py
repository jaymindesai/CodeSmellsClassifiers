import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from skfeature.function.statistical_based import CFS
from skfeature.function.wrapper import decision_tree_forward as dtf
from skfeature.function.wrapper import svm_forward as svmf

import context

from collections import defaultdict
from time import time

df = pd.read_csv('../_data/cl-data-class.csv')
K = {'def': 5, 'cfs': 4, 'dtf': 5, 'svmf': 7}

# df = pd.read_csv('../_data/cl-god-class.csv')
# K = {'def': 5, 'cfs': 6, 'dtf': 4, 'svmf': 5}

# df = pd.read_csv('../_data/ml-feature-envy.csv')
# K = {'def': 6, 'cfs': 6, 'dtf': 5, 'svmf': 6}

# df = pd.read_csv('../_data/ml-long-method.csv')
# K = {'def': 6, 'cfs': 5, 'dtf': 4, 'svmf': 5}

skfolds = StratifiedKFold(n_splits=5, random_state=0)

classifiers = {'rand': DummyClassifier(strategy='uniform', random_state=0),
               'oner': DecisionTreeClassifier(max_depth=1, random_state=0),
               'cart': DecisionTreeClassifier(random_state=0),
               'nb': GaussianNB(),
               'rf': RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
               'svm': SVC(kernel='rbf', gamma='scale', random_state=0)}

accuracy = defaultdict(list)
f_score = defaultdict(list)
pct_dth = defaultdict(list)
times = defaultdict(list)

# use_clfs = {'oner'}
use_clfs = {'rand', 'oner', 'cart', 'nb', 'rf', 'svm'}
# use_fsrs = {'def', 'dtf'}
use_fsrs = {'def', 'cfs', 'dtf', 'svmf'}
runs = 5

for run in range(runs):
    print('')
    print('RUN', run + 1)
    print('')

    df = shuffle(df, random_state=0)
    y = df['SMELLS']
    X = df.drop(columns=['SMELLS'])

    fold = 0
    for train_index, test_index in skfolds.split(X, y):
        print(fold + 1)

        # Prepare the data

        X_temp_def = X.iloc[train_index]
        y_temp_def = y.iloc[train_index]

        X_test_def = X.iloc[test_index]
        y_test_def = y.iloc[test_index]

        X_smote, y_smote = SMOTE(random_state=0).fit_resample(X_temp_def, y_temp_def)

        X_temp = pd.DataFrame(X_smote)
        y_temp = pd.Series(y_smote)

        rows, cols = X_temp.shape
        num_feats = int(cols ** 0.5)

        print(cols)
        print(X.shape[0])
        print('SMOTE', rows + X_test_def.shape[0])

        # feature_selectors = {'def': None,
        #                      'cfs': CFS.cfs(X_temp.values, y_temp.values),
        #                      'dtf': dtf.decision_tree_forward(X_temp.values, y_temp.values, num_feats),
        #                      'svmf': svmf.svm_forward(X_temp.values, y_temp.values, num_feats)}

        feature_selectors = {'def': None}

        for clf_name, clf in classifiers.items():
            if clf_name in use_clfs:
                if clf_name in {'rand'}:
                    use_features = {'def': feature_selectors['def']}
                    clust = False
                else:
                    use_features = feature_selectors
                    clust = True

                for fsr_name, sel_feats in use_features.items():
                    if fsr_name in use_fsrs:
                        X_test = X_test_def
                        y_test = y_test_def

                        if sel_feats is not None:
                            X_train = X_temp.iloc[:, sel_feats]
                            X_test = X_test.iloc[:, sel_feats]
                        else:
                            X_train = X_temp
                            X_test = X_test

                        y_train = y_temp

                        # Unclustered Classification

                        start = time()

                        cloned_clf = clone(clf)
                        cloned_clf.fit(X_train, y_train)

                        y_pred = cloned_clf.predict(X_test)

                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                        end = time()

                        accuracy['{}-{}{}'.format(clf_name, fsr_name, '')].append(context.acc(tn, fp, fn, tp))
                        f_score['{}-{}{}'.format(clf_name, fsr_name, '')].append(context.f_score(fp, fn, tp))
                        pct_dth['{}-{}{}'.format(clf_name, fsr_name, '')].append(context.pct_dth(tn, fp, fn, tp))
                        times['{}-{}{}'.format(clf_name, fsr_name, '')].append(end - start)

                        # Clustered Classification

                        if clust:
                            start = time()

                            km = KMeans(n_clusters=K[fsr_name], random_state=0, precompute_distances=True,
                                        verbose=0).fit(X_train)

                            X_train = X_train.assign(cluster=km.labels_)
                            X_test = X_test.assign(cluster=km.predict(X_test))

                            y_true = []
                            y_pred = []

                            clusters = X_test['cluster'].value_counts().index
                            for c in clusters:
                                X_train_new = X_train[X_train['cluster'] == c]
                                y_train_new = y_train[X_train_new.index.values]
                                X_test_new = X_test[X_test['cluster'] == c]
                                y_test_new = y_test[X_test_new.index.values]

                                y_true.extend(y_test_new.values)

                                labels_train = set(y_train_new)
                                if len(labels_train) == 1:
                                    p = [labels_train.pop()] * len(X_test_new)
                                    y_pred.extend(p)
                                else:
                                    cloned_clf = clone(clf)
                                    cloned_clf.fit(X_train_new, y_train_new)
                                    y_pred.extend(cloned_clf.predict(X_test_new))

                            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                            end = time()

                            accuracy['{}-{}{}'.format(clf_name, fsr_name, '-clust')].append(context.acc(tn, fp, fn, tp))
                            f_score['{}-{}{}'.format(clf_name, fsr_name, '-clust')].append(context.f_score(fp, fn, tp))
                            pct_dth['{}-{}{}'.format(clf_name, fsr_name, '-clust')].append(
                                context.pct_dth(tn, fp, fn, tp))
                            times['{}-{}{}'.format(clf_name, fsr_name, '-clust')].append(end - start)
        fold += 1

print('----------')

results = {'accuracy': accuracy, 'f_score': f_score, 'pct_dth': pct_dth, 'times': times}

for metric, result in results.items():
    print(metric)
    print('')
    for k, v in result.items():
        # print(k)
        # v = map(str, v)
        v.sort()
        # print(' '.join(v))
        if metric == 'pct_dth':
            print(k, '', round(1 - v[(0 + len(v)) // 2], 2))
        else:
            print(k, '', round(v[(0 + len(v)) // 2], 2))
        print('')
    print('')
    print('----------')
    print('')
