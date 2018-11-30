import os
import pandas
import sys
import time

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

root_path = os.path.dirname(os.path.abspath(__file__)).split('fs_clfs')[0]

if root_path not in sys.path:
    sys.path.append(str(root_path))

import context

from skfeature.function.statistical_based import CFS
from skfeature.function.wrapper import decision_tree_forward as dtf
from skfeature.function.wrapper import svm_forward as svmf

use_clfs = ['rand', 'nb', 'svm', 'oner', 'cart', 'rf']

for file in context.FILES:

    # Retrieve the file name for reporting
    file_name = file.split('/')[-1].split('.')[0]

    # Generate a DataFrame from the csv file
    # Drop the two symbolic columns if present
    data_frame = pandas.read_csv(file)
    data_frame.drop(columns=context.SYM_COLS, inplace=True, errors='ignore')

    # Initialize the stratified k-folds
    skfolds = StratifiedKFold(n_splits=5, random_state=0)

    # Initialize the classifiers
    classifiers = {'rand': DummyClassifier(strategy='uniform', random_state=0),
                   'oner': DecisionTreeClassifier(max_depth=1, random_state=0),
                   'cart': DecisionTreeClassifier(random_state=0),
                   'nb': GaussianNB(),
                   'rf': RandomForestClassifier(random_state=0),
                   'svm': SVC(random_state=0)}

    # Set up a dictionary to record performance metrics for each classifier across runs/folds
    metrics = {'rand':
                   {'def':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
               'oner':
                   {'def':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'cfs':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'dtf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'svmf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
               'cart':
                   {'def':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'cfs':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'dtf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'svmf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
               'nb':
                   {'def':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'cfs':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'dtf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'svmf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
               'rf':
                   {'def':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'cfs':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'dtf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'svmf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
               'svm':
                   {'def':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'cfs':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'dtf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                    'svmf':
                        {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}}}

    smote_metrics = {'rand':
                         {'def':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                     'oner':
                         {'def':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'cfs':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'dtf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'svmf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                     'cart':
                         {'def':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'cfs':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'dtf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'svmf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                     'nb':
                         {'def':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'cfs':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'dtf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'svmf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                     'rf':
                         {'def':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'cfs':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'dtf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'svmf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                     'svm':
                         {'def':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'cfs':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'dtf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                          'svmf':
                              {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}}}

    adasyn_metrics = {'rand':
                          {'def':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                      'oner':
                          {'def':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'cfs':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'dtf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'svmf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                      'cart':
                          {'def':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'cfs':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'dtf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'svmf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                      'nb':
                          {'def':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'cfs':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'dtf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'svmf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                      'rf':
                          {'def':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'cfs':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'dtf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'svmf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}},
                      'svm':
                          {'def':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'cfs':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'dtf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
                           'svmf':
                               {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}}}

    # Run the 5-fold cross-validation for 5 runs, shuffling each run
    for run in range(5):

        # Shuffle the DataFrame
        data_frame = shuffle(data_frame, random_state=0)

        # Store the labels, translating True -> 1 and False -> 0
        labels = data_frame['SMELLS'].apply(lambda x: 1 if x else 0)

        # Remove the labels from the DataFrame
        unlabeled_data = data_frame.drop(columns=['SMELLS'])

        fold = 0

        for train_indices, test_indices in skfolds.split(unlabeled_data, labels):

            # Clone the classifiers
            cloned_classifiers = {'rand':
                                      {'def': clone(classifiers['rand'])},
                                  'oner':
                                      {'def': clone(classifiers['oner']),
                                       'cfs': clone(classifiers['oner']),
                                       'dtf': clone(classifiers['oner']),
                                       'svmf': clone(classifiers['oner'])},
                                  'cart':
                                      {'def': clone(classifiers['cart']),
                                       'cfs': clone(classifiers['cart']),
                                       'dtf': clone(classifiers['cart']),
                                       'svmf': clone(classifiers['cart'])},
                                  'nb':
                                      {'def': clone(classifiers['nb']),
                                       'cfs': clone(classifiers['nb']),
                                       'dtf': clone(classifiers['nb']),
                                       'svmf': clone(classifiers['nb'])},
                                  'rf':
                                      {'def': clone(classifiers['rf']),
                                       'cfs': clone(classifiers['rf']),
                                       'dtf': clone(classifiers['rf']),
                                       'svmf': clone(classifiers['rf'])},
                                  'svm':
                                      {'def': clone(classifiers['svm']),
                                       'cfs': clone(classifiers['svm']),
                                       'dtf': clone(classifiers['svm']),
                                       'svmf': clone(classifiers['svm'])}}

            # Retrieve the training data and training labels for the fold
            train_data = unlabeled_data.iloc[train_indices]
            train_labels = labels.iloc[train_indices]

            # Retrieve the testing data and testing labels for the fold
            test_data = unlabeled_data.iloc[test_indices]
            test_labels = labels.iloc[test_indices]

            # print(f'\n----- {file_name}-{run}-{fold} -----')

            cfs_start = time.perf_counter()
            cfs_feats = CFS.cfs(train_data.values, train_labels.values)
            cfs_end = time.perf_counter()
            cfs_time = cfs_end - cfs_start

            cfs_train_data = train_data.iloc[:, cfs_feats]
            cfs_test_data = test_data.iloc[:, cfs_feats]
            # print('\nCFS', list(cfs_train_data), '\n')

            rows, cols = train_data.shape
            num_feats = int(cols ** 0.5)

            dtf_start = time.perf_counter()
            dtf_feats = dtf.decision_tree_forward(train_data.values, train_labels.values, num_feats)
            dtf_end = time.perf_counter()
            dtf_time = dtf_end - dtf_start

            dtf_train_data = train_data.iloc[:, dtf_feats]
            dtf_test_data = test_data.iloc[:, dtf_feats]
            # print('DTF', list(dtf_train_data), '\n')

            svmf_start = time.perf_counter()
            svmf_feats = svmf.svm_forward(train_data.values, train_labels.values, num_feats)
            svmf_end = time.perf_counter()
            svmf_time = svmf_end - svmf_start

            svmf_train_data = train_data.iloc[:, svmf_feats]
            svmf_test_data = test_data.iloc[:, svmf_feats]
            # print('SVMF', list(svmf_train_data), '\n')

            # print(f'{file_name.replace("_", "")},{cfs_time},{dtf_time},{svmf_time}')

            # SMOTE
            smote_train_data, smote_train_labels = SMOTE().fit_resample(train_data, train_labels)

            smote_cfs_train_data, smote_cfs_train_labels = SMOTE().fit_resample(cfs_train_data, train_labels)

            smote_dtf_train_data, smote_dtf_test_data = SMOTE().fit_resample(dtf_train_data, train_labels)

            smote_svmf_train_data, smote_svmf_test_data = SMOTE().fit_resample(svmf_train_data, train_labels)

            # ADASYN
            adasyn_train_data, adasyn_test_data = ADASYN().fit_resample(train_data, train_labels)

            adasyn_cfs_train_data, adasyn_cfs_test_data = ADASYN().fit_resample(cfs_train_data, train_labels)

            adasyn_dtf_train_data, adasyn_dtf_test_data = ADASYN().fit_resample(dtf_train_data, train_labels)

            adasyn_svmf_train_data, adasyn_svmf_test_data = ADASYN().fit_resample(svmf_train_data, train_labels)

            # Train the classifiers for the fold
            for clf in cloned_classifiers:
                if clf in use_clfs:
                    for fs in cloned_classifiers[clf]:
                        if fs == 'def':
                            cloned_classifiers[clf][fs].fit(train_data, train_labels)
                        elif fs == 'cfs':
                            cloned_classifiers[clf][fs].fit(cfs_train_data, train_labels)
                        elif fs == 'dtf':
                            cloned_classifiers[clf][fs].fit(dtf_train_data, train_labels)
                        elif fs == 'svmf':
                            cloned_classifiers[clf][fs].fit(svmf_train_data, train_labels)

            predictions = {'rand':
                               {'def': None},
                           'oner':
                               {'def': None, 'cfs': None, 'dtf': None, 'svmf': None},
                           'cart':
                               {'def': None, 'cfs': None, 'dtf': None, 'svmf': None},
                           'nb':
                               {'def': None, 'cfs': None, 'dtf': None, 'svmf': None},
                           'rf':
                               {'def': None, 'cfs': None, 'dtf': None, 'svmf': None},
                           'svm':
                               {'def': None, 'cfs': None, 'dtf': None, 'svmf': None}}

            # Test the classifiers for the fold
            for clf in predictions:
                if clf in use_clfs:
                    for fs in predictions[clf]:
                        if fs == 'def':
                            predictions[clf][fs] = cloned_classifiers[clf][fs].predict(test_data)
                        elif fs == 'cfs':
                            predictions[clf][fs] = cloned_classifiers[clf][fs].predict(cfs_test_data)
                        elif fs == 'dtf':
                            predictions[clf][fs] = cloned_classifiers[clf][fs].predict(dtf_test_data)
                        elif fs == 'svmf':
                            predictions[clf][fs] = cloned_classifiers[clf][fs].predict(svmf_test_data)

            matrix_metrics = {'rand':
                                  {'def':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None}},
                              'oner':
                                  {'def':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'cfs':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'dtf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'svmf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None}},
                              'cart':
                                  {'def':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'cfs':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'dtf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'svmf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None}},
                              'nb':
                                  {'def':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'cfs':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'dtf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'svmf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None}},
                              'rf':
                                  {'def':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'cfs':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'dtf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'svmf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None}},
                              'svm':
                                  {'def':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'cfs':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'dtf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                                   'svmf':
                                       {'tn': None, 'fp': None, 'fn': None, 'tp': None}}}

            # Retrieve the confusion matrix metrics for the classifiers
            for clf in matrix_metrics:
                if clf in use_clfs:
                    for fs in matrix_metrics[clf]:
                        matrix_metrics[clf][fs]['tn'], \
                        matrix_metrics[clf][fs]['fp'], \
                        matrix_metrics[clf][fs]['fn'], \
                        matrix_metrics[clf][fs]['tp'] = confusion_matrix(test_labels, predictions[clf][fs]).ravel()

            # Calculate the Accuracy, F-Score, Kappa, Percent Distance to Heaven, and Informedness metrics
            for clf in metrics:
                if clf in use_clfs:
                    for fs in metrics[clf]:
                        metrics[clf][fs]['acc'].append(context.acc(matrix_metrics[clf][fs]['tn'],
                                                                   matrix_metrics[clf][fs]['fp'],
                                                                   matrix_metrics[clf][fs]['fn'],
                                                                   matrix_metrics[clf][fs]['tp']))

                        metrics[clf][fs]['f_score'].append(context.f_score(matrix_metrics[clf][fs]['fp'],
                                                                           matrix_metrics[clf][fs]['fn'],
                                                                           matrix_metrics[clf][fs]['tp']))

                        metrics[clf][fs]['kappa'].append(context.kappa(matrix_metrics[clf][fs]['tn'],
                                                                       matrix_metrics[clf][fs]['fp'],
                                                                       matrix_metrics[clf][fs]['fn'],
                                                                       matrix_metrics[clf][fs]['tp'],
                                                                       matrix_metrics['rand']['def']['tn'],
                                                                       matrix_metrics['rand']['def']['fp'],
                                                                       matrix_metrics['rand']['def']['fn'],
                                                                       matrix_metrics['rand']['def']['tp']))

                        metrics[clf][fs]['inform'].append(context.inform(matrix_metrics[clf][fs]['tn'],
                                                                         matrix_metrics[clf][fs]['fp'],
                                                                         matrix_metrics[clf][fs]['fn'],
                                                                         matrix_metrics[clf][fs]['tp']))

                        metrics[clf][fs]['pct_dth'].append(context.pct_dth(matrix_metrics[clf][fs]['tn'],
                                                                           matrix_metrics[clf][fs]['fp'],
                                                                           matrix_metrics[clf][fs]['fn'],
                                                                           matrix_metrics[clf][fs]['tp']))

            fold += 1

    # print(f'----- {file_name} -----\n')
    # for clf in metrics:
    #     for fs in metrics[clf]:
    #         print(f'-- {clf}-{fs} --\n')
    #         for metric in metrics[clf][fs]:
    #             print(metric)
    #             print([round(x, 2) for x in metrics[clf][fs][metric]], '\n')

    # for clf in metrics:
    #         for fs in metrics[clf]:
    #             for metric in metrics[clf][fs]:
    #                 with open(f'{context.ROOT}/_output/{file_name}/{file_name}-{metric}.txt', 'a+') as output_file:
    #                     output_file.write(f'{clf}-{fs}\n')
    #                     for value in metrics[clf][fs][metric]:
    #                         output_file.write(f'{value} ')
    #                     output_file.write('\n\n')

    # for clf in metrics:
    #     for metric in metrics[clf]:
    #         with open(f'{context.ROOT}/_output/{file_name}/{file_name}-{metric}.txt', 'a+') as output_file:
    #             output_file.write(f'{clf}\n')
    #             for index, value in enumerate(metrics[clf][metric]):
    #                 output_file.write(f'{value} ')
    #             output_file.write('\n\n')
