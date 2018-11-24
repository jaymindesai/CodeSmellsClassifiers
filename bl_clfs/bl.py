import os
import pandas
import sys

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

root_path = os.path.dirname(os.path.abspath(__file__)).split('bl_clfs')[0]

if root_path not in sys.path:
    sys.path.append(str(root_path))

import context

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
                   'nb': GaussianNB(), 'rf': RandomForestClassifier(random_state=0),
                   'svm': SVC(random_state=0)}

    # Set up a dictionary to record performance metrics for each classifier across runs/folds
    metrics = {'rand':
                   {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
               'oner':
                   {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
               'cart':
                   {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
               'nb':
                   {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
               'rf':
                   {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []},
               'svm':
                   {'acc': [], 'f_score': [], 'kappa': [], 'inform': [], 'pct_dth': []}}

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
            cloned_classifiers = {'rand': clone(classifiers['rand']),
                                  'oner': clone(classifiers['oner']),
                                  'cart': clone(classifiers['cart']),
                                  'nb': clone(classifiers['nb']),
                                  'rf': clone(classifiers['rf']),
                                  'svm': clone(classifiers['svm'])}

            # Retrieve the training data and training labels for the fold
            train_data = unlabeled_data.iloc[train_indices]
            train_labels = labels.iloc[train_indices]

            # Retrieve the testing data and testing labels for the fold
            test_data = unlabeled_data.iloc[test_indices]
            test_labels = labels.iloc[test_indices]

            # Train the classifiers for the fold
            for clf in cloned_classifiers:
                cloned_classifiers[clf].fit(train_data, train_labels)

            predictions = {'rand': None, 'oner': None, 'cart': None, 'nb': None, 'rf': None, 'svm': None}

            # Test the classifiers for the fold
            for clf in predictions:
                predictions[clf] = cloned_classifiers[clf].predict(test_data)

            matrix_metrics = {'rand':
                                  {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                              'oner':
                                  {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                              'cart':
                                  {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                              'nb':
                                  {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                              'rf':
                                  {'tn': None, 'fp': None, 'fn': None, 'tp': None},
                              'svm':
                                  {'tn': None, 'fp': None, 'fn': None, 'tp': None}}

            # Retrieve the confusion matrix metrics for the classifiers
            for clf in matrix_metrics:

                matrix_metrics[clf]['tn'], \
                matrix_metrics[clf]['fp'], \
                matrix_metrics[clf]['fn'], \
                matrix_metrics[clf]['tp'] = confusion_matrix(test_labels, predictions[clf]).ravel()

            # Calculate the Accuracy, F-Score, Kappa, Percent Distance to Heaven, and Informedness metrics
            for clf in metrics:

                metrics[clf]['acc'].append(context.acc(matrix_metrics[clf]['tn'],
                                                       matrix_metrics[clf]['fp'],
                                                       matrix_metrics[clf]['fn'],
                                                       matrix_metrics[clf]['tp']))

                metrics[clf]['f_score'].append(context.f_score(matrix_metrics[clf]['fp'],
                                                               matrix_metrics[clf]['fn'],
                                                               matrix_metrics[clf]['tp']))

                metrics[clf]['kappa'].append(context.kappa(matrix_metrics[clf]['tn'],
                                                           matrix_metrics[clf]['fp'],
                                                           matrix_metrics[clf]['fn'],
                                                           matrix_metrics[clf]['tp'],
                                                           matrix_metrics['rand']['tn'],
                                                           matrix_metrics['rand']['fp'],
                                                           matrix_metrics['rand']['fn'],
                                                           matrix_metrics['rand']['tp']))

                metrics[clf]['inform'].append(context.inform(matrix_metrics[clf]['tn'],
                                                             matrix_metrics[clf]['fp'],
                                                             matrix_metrics[clf]['fn'],
                                                             matrix_metrics[clf]['tp']))

                metrics[clf]['pct_dth'].append(context.pct_dth(matrix_metrics[clf]['tn'],
                                                               matrix_metrics[clf]['fp'],
                                                               matrix_metrics[clf]['fn'],
                                                               matrix_metrics[clf]['tp']))


            fold += 1

    # print(f'----- {file_name} -----\n')
    # for clf in classifiers:
    #     if clf != 'rand':
    #         print(f'-- {clf} --\n')
    #         for metric in metrics[clf]:
    #             print(metric)
    #             print([round(x, 2) for x in metrics[clf][metric]], '\n')

    for clf in metrics:
        if clf != 'rand':
            for metric in metrics[clf]:
                with open(f'{context.ROOT}/_output/{file_name}/{file_name}-{metric}.txt', 'a+') as output_file:
                    output_file.write(f'{clf}\n')
                    for index, value in enumerate(metrics[clf][metric]):
                        output_file.write(f'{value} ')
                    output_file.write('\n\n')
